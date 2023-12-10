import logging
from typing import List, Dict, Tuple, Any
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_ab_params(
        y: float, yp: float, t: float, tp: float, sig_z: float, sig_y: float
):
    """ Get a, b parameters used in the analytical solution """

    sig_t = np.sqrt(t * sig_z ** 2 + sig_y ** 2)
    sig_tp = np.sqrt(tp * sig_z ** 2 + sig_y ** 2)

    den = ((sig_t ** 2) * (sig_tp ** 2) -
           np.minimum(t, tp) ** 2 * (sig_z ** 4))

    b_num = (y * t * sig_tp ** 2 +
             yp * tp * sig_t ** 2 -
             np.minimum(t, tp) * (sig_z ** 2) * (y * tp + yp * t))

    a_num = (t ** 2 * sig_tp ** 2 +
             tp ** 2 * sig_t ** 2 -
             2 * np.minimum(t, tp) * sig_z ** 2 * t * tp)

    a = a_num / den
    b = b_num / den

    return a, b


def truncated_normal_dist(x, mu, sig):
    """ Truncated normal distribution that is set to 0 for x < 0 """

    rv = sc.norm(loc=mu, scale=sig)
    norm = 1 - rv.cdf(0)

    return (x >= 0) * rv.pdf(x) / norm


def growth_analytical_exp_prior(
        y: np.array, yp: np.array, t: np.array, tp: np.array,
        sig_z: float, sig_y: float, lam: float
):
    """ Extend the previous function to the case of having multiple trees """

    assert len(y) == len(yp) == len(t) == len(tp)
    assert np.all(tp < t), "tp < t should be fulfilled"

    a, b = get_ab_params(y, yp, t, tp, sig_z, sig_y)

    mu = (b.sum() - lam) / a.sum()
    sig = np.sqrt(1 / a.sum())

    return dict(mu=mu, sig=sig)


def generate_data(
        w: float = 1.4,
        sig_z: float = .2,
        sig_y: float = .4,
        nmax: int = 8,
        tmin: int = 2,
        tmax: int = 10,
        seed: int = 12,
        **kwargs
):
    """ Generate nmax trajectories from a local level model with drift w

        y_t = z_t + epsilon_t
        z_t = z_{t-1} + w + eta_t
    """

    assert 0 < tmin < tmax, "tmin < tmax should be fulfilled"

    np.random.seed(seed)

    epsilon = np.random.randn(nmax, tmax) * sig_y
    eta = np.random.randn(nmax, tmax) * sig_z
    eta[:, 0] = 0

    # shape: (nmax, tmax)
    drift = np.arange(tmax) * np.ones(shape=(nmax, 1)) * w
    ys = drift + epsilon + eta.cumsum(axis=1)

    # take 2 time indices from every trajectory; shape = (nmax, 2)
    ts_obs = np.stack([np.random.choice(range(tmin, tmax),
                                        size=(2,),
                                        replace=False) for _ in range(nmax)])
    # sort the indices for every row
    ts_obs = np.sort(ts_obs)

    # keep the ys from the selected time stamps; shape = (nmax, 2)
    ys_obs = np.stack([y[t] for t, y in zip(ts_obs, ys)])

    return ys, ts_obs, ys_obs


def plot_results(
        ns: List[int],
        results: List[Dict[str, float]],
        results_mcmc: List[Tuple[Any, Any]],
        xmin: float = -.2,
        xmax: float = .6
):
    rows = len(ns) // 2

    fig = plt.figure(figsize=(15, 4 * rows))

    for idx, n in enumerate(ns):
        res_true = results[idx]

        w_samples = results_mcmc[idx][0][0].numpy().reshape(-1)

        ax = fig.add_subplot(rows, 2, idx + 1)
        x = np.linspace(xmin, xmax, 200)

        ax.plot(x, truncated_normal_dist(x, res_true['mu'], res_true['sig']),
                label=f'analytical (n={n})')

        ax.plot(x, sc.kde.gaussian_kde(w_samples)(x), label=f'MCMC (n={n})')

        ax.legend()
        ax.set_xlabel(r'$\omega$')

    return fig


def plot_data(ys: np.ndarray):
    t = np.arange(ys.shape[-1])

    fig = plt.figure(figsize=(6, 3))

    for y in ys:
        plt.plot(t, y, color='tab:blue', alpha=.3)

    plt.ylabel('y')
    plt.xlabel('t')

    return fig
