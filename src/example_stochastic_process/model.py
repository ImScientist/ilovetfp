import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

log = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR')

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float32


def cov_mat(t0, t1, sig_z, sig_y):
    """ Cov 2x2 matrix for two observations from a local level process
    (t0 < t1)
    """

    cov = (sig_z ** 2 * np.array([[t0, t0], [t0, t1]]) +
           sig_y ** 2 * np.eye(2))

    return cov


def build_model(
        ts_obs: np.ndarray,  # (n,2,)
        ys_obs: np.ndarray,  # (n,2,)
        sig_z: float,
        sig_y: float,
        lam_0: float
):
    assert ts_obs.shape[0] == ys_obs.shape[0]
    assert ts_obs.shape[1] == ys_obs.shape[1] == 2

    # shape = (n, 2, 2)
    covs = np.stack([cov_mat(*t, sig_z, sig_y) for t in ts_obs])
    covs = tf.cast(covs, dtype)

    # shape = (n, 2)
    ts_obs = tf.cast(ts_obs, dtype)
    ys_obs = tf.cast(ys_obs, dtype)

    # shape = (1, 1)
    lam_0 = tf.cast([[lam_0]], dtype)

    model = tfd.JointDistributionSequential([
        tfd.Independent(
            distribution=tfd.Exponential(rate=lam_0, force_probs_to_zero_outside_support=True),
            reinterpreted_batch_ndims=2,
            name='w'),
        lambda w: tfd.Independent(
            distribution=tfd.MultivariateNormalFullCovariance(
                loc=w * ts_obs,
                covariance_matrix=covs),
            reinterpreted_batch_ndims=1,
            name='likelihood')
    ])

    model_log_prob_fn = lambda *w: model.log_prob(w + (ys_obs,))

    return model, model_log_prob_fn
