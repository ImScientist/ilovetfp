import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Literal

logger = logging.getLogger(__name__)

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float32


def weights_prior(name: str, lam):
    """ Weights prior with event_shape = (1, 1) """

    return tfd.Independent(
        distribution=tfd.Exponential(
            rate=tf.constant([[lam]], dtype=dtype),
            force_probs_to_zero_outside_support=True),
        reinterpreted_batch_ndims=2,
        name=name)


def build_model(lam_0: float, y_sig: float, x: np.array, y: np.array):
    """ Build model and log-likelihood fn

    p(y,w|x)       = p(y|w,x) * p(w|x)
                   = p(y|w,x) * p(w)

    p(w)           = lam * exp( -lam * w )

    p(y|w,x)       = Prod_{i} p_i(y_i|w,x_i)

    p_i(y_i|w,x_i) = Normal(y_i - w * x_i, y_sig**2)
    """

    sig = tf.cast(y_sig, dtype)
    x = tf.cast(x, dtype)  # shape = (data points, 1)
    y = tf.cast(y, dtype)  # shape = (data points, 1)

    model = tfd.JointDistributionSequential([
        weights_prior(name='w', lam=lam_0),
        lambda w: tfd.Independent(
            distribution=tfd.Normal(loc=tf.matmul(x, w), scale=sig),
            reinterpreted_batch_ndims=2,
            name='y')
    ])

    model_log_prob_fn = lambda *x_: model.log_prob(x_ + (y,))

    return model, model_log_prob_fn


def build_surrogate_posterior(
        distribution: Literal['truncated_normal', 'log_normal']
):
    """ Build surrogate posterior with trainable parameters """

    if distribution == 'log_normal':
        posterior = tfd.JointDistributionSequential([
            tfd.Independent(
                distribution=tfd.LogNormal(
                    loc=tf.Variable(
                        initial_value=tf.random.normal(shape=(1, 1), stddev=.01),
                        name='w_loc',
                        dtype=dtype),
                    scale=tfp.util.TransformedVariable(
                        initial_value=tf.constant([[1.]]),
                        bijector=tfb.Softplus(),
                        name='w_scale',
                        dtype=dtype)),
                reinterpreted_batch_ndims=2,
                name='w')
        ])

    else:
        posterior = tfd.JointDistributionSequential([
            tfd.Independent(
                distribution=tfd.TruncatedNormal(
                    loc=tf.Variable(
                        initial_value=tf.random.normal(shape=(1, 1), stddev=.01),
                        name='w_loc',
                        dtype=dtype),
                    scale=tfp.util.TransformedVariable(
                        initial_value=tf.constant([[1.]]),
                        bijector=tfb.Softplus(),
                        name='w_scale',
                        dtype=dtype),
                    low=0.,
                    high=10.),
                reinterpreted_batch_ndims=2,
                name='w')
        ])

    return posterior
