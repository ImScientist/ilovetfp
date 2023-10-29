import logging
import tensorflow as tf
import tensorflow_probability as tfp

logger = logging.getLogger(__name__)

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float32


def weights_hyperprior_trainable(name: str, n: int):
    """ Weights hyperprior with trainable parameters and
    event_shape = (n, 1) """

    return tfd.Independent(
        distribution=tfd.LogNormal(
            loc=tf.Variable(
                initial_value=tf.random.uniform((n, 1), minval=-.2, maxval=.2, dtype=dtype),
                name=f'{name}_loc',
                dtype=dtype),
            scale=tfp.util.TransformedVariable(
                initial_value=tf.constant([[1.]] * n, dtype=dtype),
                bijector=tfb.Softplus(),
                name=f'{name}_scale')),
        reinterpreted_batch_ndims=2,
        name=name)


def weights_prior_trainable(name: str, n: int):
    """ Weights prior with trainable parameters and
    event_shape = (n, 1) """

    return tfd.Independent(
        distribution=tfd.Normal(
            loc=tf.Variable(
                initial_value=tf.random.uniform((n, 1), minval=-.2, maxval=.2, dtype=dtype),
                name=f'{name}_loc',
                dtype=dtype),
            scale=tfp.util.TransformedVariable(
                initial_value=tf.constant([[1.]] * n, dtype=dtype),
                bijector=tfb.Softplus(),
                name=f'{name}_scale')),
        reinterpreted_batch_ndims=2,
        name=name)


def build_surrogate_posterior(weight_dim: int):
    """ Build surrogate posterior with trainable parameters """

    args = {'n': weight_dim}

    prior = tfd.JointDistributionSequential([
        weights_hyperprior_trainable(name='lam_a', **args),
        weights_hyperprior_trainable(name='lam_b', **args),
        weights_prior_trainable(name='a', **args),
        weights_prior_trainable(name='b', **args)
    ])

    return prior
