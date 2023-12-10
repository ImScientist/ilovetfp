import logging
import os

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from .model import build_model
from .mcmc import run_chain
from .misc import plot_results, plot_data, generate_data, growth_analytical_exp_prior

logger = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR')

opj = os.path.join
tfb = tfp.bijectors
dtype = tf.float32


def main(save_dir: str):
    """ Run experiment """

    os.makedirs(save_dir, exist_ok=True)

    ns = [1, 2, 4, 8]

    params = dict(
        w=.5,
        sig_z=.2,
        sig_y=.4,
        lam_0=70,
        nmax=max(ns),
        tmin=1,
        tmax=5,
        seed=12)

    ys, ts_obs, ys_obs = generate_data(**params)

    # ##### Analytical solution
    results = []

    for n in ns:
        res = growth_analytical_exp_prior(
            y=ys_obs[:n, 1], yp=ys_obs[:n, 0],
            t=ts_obs[:n, 1], tp=ts_obs[:n, 0],
            sig_z=params['sig_z'],
            sig_y=params['sig_y'],
            lam=params['lam_0']
        )
        results.append(res)

    df_true = pd.DataFrame(results, index=ns)
    df_true.to_csv(opj(save_dir, 'analytical_results.csv'), index=False)

    # ##### MCMC
    results_mcmc = []

    for n in ns:
        logger.info(f'Trajectories: {n}')
        nchain = 8

        model, target_log_prob_fn = build_model(
            ts_obs=ts_obs[:n], ys_obs=ys_obs[:n],
            sig_z=params['sig_z'], sig_y=params['sig_y'], lam_0=params['lam_0'])

        w0, _ = model.sample(nchain)

        init_state = [w0]
        step_size = [tf.cast(i, dtype=dtype) for i in [.01]]
        unconstraining_bijectors = [tfb.Identity()]

        samples, sampler_stat = run_chain(
            init_state, step_size, target_log_prob_fn, unconstraining_bijectors,
            num_steps=5_000, burnin=500)

        results_mcmc.append((samples, sampler_stat))

    fig = plot_results(ns=ns, results=results, results_mcmc=results_mcmc)
    fig.savefig(opj(save_dir, 'posteriors.png'), dpi=200, bbox_inches='tight')

    fig = plot_data(ys=ys)
    fig.savefig(opj(save_dir, 'data.png'), dpi=200, bbox_inches='tight')
