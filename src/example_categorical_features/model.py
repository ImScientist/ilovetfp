import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import (
    prefer_static as ps,
    dtype_util,
    tensor_util,
    reparameterization,
    parameter_properties)

logger = logging.getLogger(__name__)

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float32


class CustomDist(distribution.AutoCompositeTensorDistribution):
    """ A distribution object that has only a well-defined log_prob fn """

    def __init__(
            self,
            loc,
            scale,
            counts,
            validate_args=False,
            allow_nan_stats=True,
            name='CustomDist'
    ):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype(
                [loc, scale, counts], dtype_hint=tf.float32)

            self._loc = tensor_util.convert_nonref_to_tensor(
                loc, dtype=self.dtype, name='loc')

            self._scale = tensor_util.convert_nonref_to_tensor(
                scale, dtype=self.dtype, name='scale')

            self._counts = tensor_util.convert_nonref_to_tensor(
                counts, dtype=self.dtype, name='counts')

            super(CustomDist, self).__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name)

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the scale."""
        return self._scale

    @property
    def counts(self):
        """Distribution parameter for the counts."""
        return self._counts

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            loc=parameter_properties.ParameterProperties(),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            counts=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _sample_n(self, n, seed=None):
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        counts = tf.convert_to_tensor(self.counts)

        bs_tensor = self._batch_shape_tensor(loc=loc, scale=scale, counts=counts)

        shape = ps.concat([[n], bs_tensor], axis=0)

        return tf.zeros(shape=shape, dtype=self.dtype)

    def _log_prob(self, x):
        x_mean = x[..., :1]
        x_std = x[..., 1:]
        zero = tf.constant(0, dtype=self.dtype)

        log_unnormalized = -0.5 * self.counts * (
                tf.math.squared_difference(x_mean / self.scale, self.loc / self.scale) +
                tf.math.squared_difference(x_std / self.scale, zero)
        )

        log_normalization = (
                tf.constant(0.5 * np.log(2. * np.pi), dtype=self.dtype) +
                tf.math.log(self.scale)
        )

        log_normalization *= self.counts

        return log_unnormalized - log_normalization


def weights_hyperprior(name: str, n: int):
    """ Weights hyperprior with event_shape = (n, 1) """

    return tfd.Independent(
        distribution=tfd.Gamma(
            concentration=tf.constant([[1e-3]] * n, dtype=dtype),
            rate=tf.constant([[1e-3]] * n, dtype=dtype)),
        reinterpreted_batch_ndims=2,
        name=name)


def weights_prior(name: str, n: int, lam):
    """ Weights prior with event_shape = (n, 1) """

    return tfd.Independent(
        distribution=tfd.Normal(
            loc=tf.constant([[0]] * n, dtype=dtype),
            scale=lam),
        reinterpreted_batch_ndims=2,
        name=name)


def build_model(
        x: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        counts: np.ndarray
):
    """ Build log-likelihood fn for the case of using aggregated data per variant """

    weight_dim = x.shape[1]

    x = tf.cast(x, dtype)  # shape = (samples, n)
    y_mean = tf.cast(y_mean, dtype)  # shape = (samples, 1)
    y_std = tf.cast(y_std, dtype)  # shape = (samples, 1)
    counts = tf.cast(counts.reshape(-1, 1), dtype)  # shape = (samples, 1)

    # shape = (n, 2)
    y = tf.concat([y_mean, y_std], axis=-1)

    def likelihood(a, b):
        """ Likelihood """

        return tfd.Independent(
            distribution=CustomDist(
                loc=tf.matmul(x, b),
                scale=(tf.constant(1e-5, dtype=dtype) +
                       tf.nn.softplus(tf.matmul(x, a))),
                counts=counts),
            reinterpreted_batch_ndims=2,
            name='y')

    args = {'n': weight_dim}

    model = tfd.JointDistributionSequential([
        weights_hyperprior(name='lam_a', **args),
        weights_hyperprior(name='lam_b', **args),
        lambda lam_b, lam_a: weights_prior(lam=lam_a, name='a', **args),
        lambda a, lam_b, lam_a: weights_prior(lam=lam_b, name='b', **args),
        lambda b, a, lam_b, lam_a: likelihood(a=a, b=b)
    ])

    log_prob_fn = lambda *x: model.log_prob(x + (y,))

    return model, log_prob_fn


def build_model_standard(
        x: np.ndarray,
        y: np.ndarray,
):
    """ Build log-likelihood fn """

    weight_dim = x.shape[1]

    x = tf.cast(x, dtype)  # shape = (samples, n)
    y = tf.cast(y, dtype)  # shape = (samples, 1)

    def likelihood(a, b):
        """ Likelihood """

        return tfd.Independent(
            distribution=tfd.Normal(
                loc=tf.matmul(x, b),
                scale=(tf.constant(1e-5, dtype=dtype) +
                       tf.nn.softplus(tf.matmul(x, a)))),
            reinterpreted_batch_ndims=2,
            name='y')

    args = {'n': weight_dim}

    model = tfd.JointDistributionSequential([
        weights_hyperprior(name='lam_a', **args),
        weights_hyperprior(name='lam_b', **args),
        lambda lam_b, lam_a: weights_prior(lam=lam_a, name='a', **args),
        lambda a, lam_b, lam_a: weights_prior(lam=lam_b, name='b', **args),
        lambda b, a, lam_b, lam_a: likelihood(a=a, b=b)
    ])

    log_prob_fn = lambda *x: model.log_prob(x + (y,))

    return model, log_prob_fn
