import itertools
import numpy as np
import pandas as pd


def generate_data(
        samples: list[int],
        cardinalities: list[int],
        seed: int = 12
) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    """ Data generator

    For every variant type characterised by a set of categorical variables
    generate samples of the target variable. The mean and std of the target
    are specific for the variant type.

    Parameters
    ----------
    samples: number of samples per variant
    cardinalities: cardinality of every categorical feature;
        The number of categorical features is equal to `len(cardinalities)`;
        We generate samples for every combination of categorical features;
    seed:

    Return
    ------
    df: data
    a: weights of the features that contribute to the std of the target
    b: weights of the features that contribute to the mean of the target
    feature_names
    """

    assert np.prod(cardinalities) == len(samples)

    np.random.seed(seed)

    # weights: weight_{ij} = b[i][j] or a[i][j]
    a = dict()
    b = dict()

    for feature_idx, cardinality in enumerate(cardinalities):
        a[feature_idx] = dict()
        b[feature_idx] = dict()

        for c in range(cardinality):
            a[feature_idx][c] = np.random.randn() * .5
            b[feature_idx][c] = np.random.gamma(shape=8, scale=1)

    all_df = []

    ranges = [range(c) for c in cardinalities]

    # Iterate through all combinations of values of the categorical features
    for s, feature_combination in zip(samples, itertools.product(*ranges)):
        df = pd.DataFrame(index=np.arange(s))
        df['y_mean'] = 0
        df['y_std'] = 0

        for feature_idx, value in enumerate(feature_combination):
            df[f'f{feature_idx}'] = value
            df['y_mean'] += b[feature_idx][value]
            df['y_std'] += a[feature_idx][value]

        all_df.append(df)

    df = pd.concat(all_df, ignore_index=True)
    df['y_std'] = 1e-5 + np.exp(df['y_std'])
    df['y'] = df['y_mean'] + df['y_std'] * np.random.randn(len(df), )
    df = df.drop(['y_mean', 'y_std'], axis=1)

    features = df.drop(['y'], axis=1).columns.tolist()

    return df, a, b, features
