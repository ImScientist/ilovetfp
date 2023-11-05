import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tabulate import tabulate
from typing import Literal

from .model import build_model, build_surrogate_posterior
from .mcmc import run_chain
from .misc import (
    generate_data,
    growth_analytical_exp_prior,
    plot_data,
    plot_results)

log = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR')

opj = os.path.join
tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float32


def train_vi(
        xs: np.ndarray,
        ys: np.ndarray,
        posterior_dist: Literal['truncated_normal', 'log_normal'],
        lam_0: float,
        y_sig: float
):
    """ Use variational inference approach to train the model """

    log.info(f'Variational inference ({len(xs)} datapoints)')

    posterior = build_surrogate_posterior(distribution=posterior_dist)

    model, target_log_prob_fn = build_model(lam_0, y_sig, xs, ys)

    optimizer = tf.optimizers.Adam(learning_rate=1e-2)

    @tf.function(jit_compile=True)
    def fit_vi():
        return tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=target_log_prob_fn,
            surrogate_posterior=posterior,
            optimizer=optimizer,
            num_steps=10_000,
            sample_size=100)

    losses = fit_vi()

    w_dist = posterior.model[0]
    variables = posterior.variables

    optimal_parameters = dict(
        mu=variables[0].numpy()[0, 0],
        sig=tfb.Softplus(1.).forward(posterior.variables[1]).numpy()[0, 0],
        mean=w_dist.mean().numpy()[0][0],
        std=w_dist.stddev().numpy()[0][0])

    return optimal_parameters


def train_mcmc(
        xs: np.ndarray,
        ys: np.ndarray,
        lam_0: float,
        y_sig: float,
        nchain: int = 4
):
    """ Use MCMC to train a model """

    log.info(f'MCMC ({len(xs)} datapoints)')

    model, target_log_prob_fn = build_model(lam_0, y_sig, xs, ys)
    w0, _ = model.sample(nchain)

    init_state = [w0]
    step_size = [tf.cast(i, dtype=dtype) for i in [.01]]

    unconstraining_bijectors = [tfb.Identity()]

    samples, sampler_stat = run_chain(
        init_state, step_size, target_log_prob_fn, unconstraining_bijectors, num_steps=5000)

    samples = samples[0].numpy().reshape(-1)

    return samples


def compare_dist_params(
        results_analytical: list[dict],
        results_vi: list[dict],
        ns: list[int]
) -> pd.DataFrame:
    """ Compare the parameters of the Truncated Normal distribution
    obtained analytically or through the VI approach """

    df_true = pd.DataFrame(results_analytical, index=ns).loc[:, ['mu', 'sig']]
    df_true.columns = pd.MultiIndex.from_tuples([('true', c) for c in df_true.columns])

    df_vi = pd.DataFrame(results_vi, index=ns).loc[:, ['mu', 'sig']]
    df_vi.columns = pd.MultiIndex.from_tuples([('vi', c) for c in df_vi.columns])

    df = pd.concat([df_true, df_vi], axis=1)
    log.info('Parameters of the Truncated Normal distribution:\n\n'
             f'{tabulate(df.round(3), headers="keys")}\n')

    return df


def compare_weight_parameters(
        w_true: float,
        results_vi: list[dict],
        results_vi_lognorm: list[dict],
        results_mcmc: list[dict],
        ns: list[int]
) -> pd.DataFrame:
    """ Compare the inferred posterior distributions of the regression weights """

    df_true = pd.DataFrame(index=ns)
    df_true['mean'] = w_true
    df_true.columns = pd.MultiIndex.from_tuples([('true', c) for c in df_true.columns])

    df_vi = pd.DataFrame(results_vi, index=ns).loc[:, ['mean', 'std']]
    df_vi.columns = pd.MultiIndex.from_tuples([('vi', c) for c in df_vi.columns])

    df_vi_ln = pd.DataFrame(results_vi_lognorm, index=ns).loc[:, ['mean', 'std']]
    df_vi_ln.columns = pd.MultiIndex.from_tuples([('vi_lognormal', c) for c in df_vi_ln.columns])

    df_mcmc = pd.DataFrame(results_mcmc, index=ns).loc[:, ['mean', 'std']]
    df_mcmc.columns = pd.MultiIndex.from_tuples([('mcmc', c) for c in df_mcmc.columns])

    df = pd.concat([df_true, df_vi, df_vi_ln, df_mcmc], axis=1)
    log.info('Inferred regression weights:\n'
             f'{tabulate(df.round(3), headers="keys")}\n')

    return df


def main(save_dir: str):
    """
    Learn the weight `w` of the linear regression
        y ~ Normal(w * x, y_sig)

    using the Bayesian inference approach.
    The prior distribution of the weight is:
        w ~ Exp(lam_0)
    """

    os.makedirs(save_dir, exist_ok=True)

    params = {
        'lam_0': 200,
        'y_sig': 3}

    ns = [2, 3, 10, 100]

    model_params, ys, xs = generate_data(n=ns[-1])

    fig = plot_data(xs=xs, ys=ys)
    fig.savefig(opj(save_dir, 'data.png'), dpi=200, bbox_inches='tight')

    results_analytical = [
        growth_analytical_exp_prior(xs[:n], ys[:n], **params)
        for n in ns]

    results_vi = [
        train_vi(xs=xs[:n], ys=ys[:n], posterior_dist='truncated_normal', **params)
        for n in ns]

    results_vi_lognorm = [
        train_vi(xs=xs[:n], ys=ys[:n], posterior_dist='log_normal', **params)
        for n in ns]

    samples_mcmc = [
        train_mcmc(xs=xs[:n], ys=ys[:n], **params)
        for n in ns]

    # Parameters of the posterior distribution of the regression weights
    df_params = compare_dist_params(
        results_analytical=results_analytical,
        results_vi=results_vi,
        ns=ns)
    df_params.columns = df_params.columns.map('_'.join)
    df_params.to_parquet(opj(save_dir, 'distribution_params.parquet'))

    # Regression weights (mean, std)
    df_weights = compare_weight_parameters(
        w_true=model_params['w'],
        results_vi=results_vi,
        results_vi_lognorm=results_vi_lognorm,
        results_mcmc=[{'mean': s.mean(), 'std': s.std()} for s in samples_mcmc],
        ns=ns)
    df_weights.columns = df_weights.columns.map('_'.join)
    df_weights.to_parquet(opj(save_dir, 'weights_params.parquet'))

    fig = plot_results(results_analytical, results_vi, samples_mcmc, ns=ns)
    fig.savefig(opj(save_dir, 'results_truncated_normal.png'), dpi=200, bbox_inches='tight')

    fig = plot_results(results_analytical, results_vi_lognorm, samples_mcmc, ns=ns)
    fig.savefig(opj(save_dir, 'results_lognormal.png'), dpi=200, bbox_inches='tight')
