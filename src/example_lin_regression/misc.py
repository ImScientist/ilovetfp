import logging
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt

npr = np.random
logger = logging.getLogger(__name__)


def truncated_normal_dist(x, mu, sig):
    """ truncated normal distribution that is set to 0 for x < 0 """

    rv = sc.norm(loc=mu, scale=sig)
    norm = 1 - rv.cdf(0)
    return (x >= 0) * rv.pdf(x) / norm


def growth_analytical_exp_prior(
        xs: np.array, ys: np.array, lam_0: float, y_sig: float
):
    """
    Analytical solution for the parameters of the
    Truncated normal distribution of the growth rate w:

        w ~ TruncNormal(mu, sig)   (w>0)
    """

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    mu_num = (ys * xs).sum() - lam_0 * y_sig ** 2
    denom = (xs ** 2).sum()

    mu = mu_num / denom
    sig = y_sig / np.sqrt(denom)

    return dict(mu=mu, sig=sig)


def generate_data(w: float = .5, ep: float = 2.1, n: int = 500):
    np.random.seed(10)

    xs = sc.expon(scale=20, loc=30).rvs((n, 1))
    ys = w * xs + npr.randn(n, 1) * ep

    params = dict(w=w, ep=ep)

    return params, ys, xs


def plot_data(xs: np.ndarray, ys: np.ndarray):
    fig = plt.figure(figsize=(10, 4))
    plt.scatter(xs[:, 0], ys[:, 0], alpha=.3)
    plt.grid()
    plt.xlabel('age [yr]')
    plt.ylabel('height')

    return fig


def plot_results(results_analytical, results_vi, results_mcmc, ns):
    rows = len(ns) // 2

    fig = plt.figure(figsize=(15, 4 * rows))

    for idx, n in enumerate(ns, 0):
        res_true = results_analytical[idx]
        res_vi = results_vi[idx]
        w_samples = results_mcmc[idx][0].numpy().reshape(-1)

        ax = fig.add_subplot(rows, 2, idx + 1)
        x = np.linspace(-.05, .55, 200)

        ax.plot(x, truncated_normal_dist(x, res_true['mu'], res_true['sig']),
                label=f'analytical (n={n})')

        ax.plot(x, truncated_normal_dist(x, res_vi['mu'], res_vi['sig']),
                label=f'variational inference (n={n})')

        ax.plot(x, sc.kde.gaussian_kde(w_samples)(x), label=f'MCMC (n={n})')

        ax.legend()
        ax.set_xlabel(r'$\omega$')

    return fig
