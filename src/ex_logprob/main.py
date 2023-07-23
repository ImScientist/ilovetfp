import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Literal
from sklearn.preprocessing import OneHotEncoder
from .misc import generate_data
from .logprob import build_log_prob_fn, build_standard_log_prob_fn
from .posterior import build_surrogate_posterior

logger = logging.getLogger(__name__)


def train(
        data: pd.DataFrame,
        oe: OneHotEncoder,
        features: list[str],
        method: Literal['standard', 'aggregated_data']
):
    """ Train a model usding aggregated data """

    data_agg = (data
                .groupby(features, as_index=False)
                .agg(y_mean=('y', 'mean'), y_std=('y', 'std'), n=('y', 'count')))

    if method == 'aggregated_data':
        x = oe.transform(data_agg[features])  # OHE features
        y_mean = data_agg[['y_mean']].values
        y_std = data_agg[['y_std']].values
        counts = data_agg[['n']].values

        log_prob_fn = build_log_prob_fn(x=x, y_mean=y_mean, y_std=y_std, counts=counts)
    else:
        x = oe.transform(data[features])
        y = data[['y']].values

        log_prob_fn = build_standard_log_prob_fn(x=x, y=y)

    surrogate_posterior = build_surrogate_posterior(weight_dim=x.shape[1])

    optimizer = tf.optimizers.Adam(learning_rate=.05)

    @tf.function(jit_compile=True)
    def fit_vi():
        return tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=log_prob_fn,
            surrogate_posterior=surrogate_posterior,
            optimizer=optimizer,
            num_steps=20_000,
            sample_size=200)

    losses = fit_vi()

    a_hat = surrogate_posterior.model[2].sample(10_000).numpy().mean(axis=0)
    b_hat = surrogate_posterior.model[3].sample(10_000).numpy().mean(axis=0)

    x = oe.transform(data_agg[features])

    data_agg['y_mean_hat'] = x @ b_hat
    data_agg['y_std_hat'] = 1e-5 + tf.nn.softplus(x @ a_hat).numpy()

    return data_agg


def main():
    """ main """

    cardinalities = [2, 4]  # generate 2 features with cardinalities 2 and 4
    samples = [1_000, 5_000, 2_000, 5_000] * 2  # 8 combinations btw the feature values

    # We should provide a number of samples for every combination of categorical features;
    assert np.prod(cardinalities) == len(samples)

    data, a, b, features = generate_data(samples=samples, cardinalities=cardinalities)

    oe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    oe.fit(data[features])

    results_agg = train(data, oe=oe, features=features, method='aggregated_data')
    results_standard = train(data, oe=oe, features=features, method='standard')

    logger.info(f'results_agg:\n{results_agg.round(2)}\n\n\n')
    logger.info(f'results_standard:\n{results_standard.round(2)}\n')
