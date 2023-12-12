from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_synthetic_data(
    n_agents, n_continuous, n_categorical=0, scaler=None, random_state=None, **kwargs
):
    """Generate synthetic data with normal distribution."""
    continuous = [f"f{i}" for i in range(n_continuous)]
    categorical = [f"cat_{i}" for i in range(n_categorical)]
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(
        rng.normal(loc=0.5, scale=1 / 3, size=(n_agents, n_continuous)),
        columns=continuous,
    )

    for cat in categorical:
        X[cat] = rng.integers(0, 2, n_agents)

    # y = rng.integers(0, 2, n_agents)
    groups = pd.Series(np.zeros(X.shape[0]), name="groups")

    if scaler is not None:
        X.loc[:, :] = scaler.transform(X)

    X = pd.concat([X, groups], axis=1)
    X = np.clip(X, 0, 1)

    return X  # , y, categorical


def get_scaler(
    n_agents=10_000,
    n_continuous=2,
    bias_factor=0,
    mean=0,
    std=1 / 4,
    random_state=None,
    **kwargs,
):
    rng = np.random.default_rng(random_state)
    groups = pd.Series(rng.binomial(1, 0.5, n_agents), name="groups")
    counts = Counter(groups)
    continuous_cols = [f"f{i}" for i in range(n_continuous)]

    # Generate the input dataset
    X_0 = pd.DataFrame(
        rng.normal(loc=mean, scale=std, size=(counts[0], n_continuous)),
        index=groups[groups == 0].index,
        columns=continuous_cols,
    )

    X_1 = pd.DataFrame(
        rng.normal(
            loc=mean + bias_factor * std, scale=std, size=(counts[1], n_continuous)
        ),
        index=groups[groups == 1].index,
        columns=continuous_cols,
    )

    X = pd.concat([X_0, X_1]).sort_index()
    return MinMaxScaler().fit(X)


def biased_data_generator(
    n_agents,
    n_continuous=2,
    bias_factor=0,
    mean=0,
    std=1 / 4,
    scaler=None,
    random_state=None,
    **kwargs,
):
    """
    Generate synthetic data.

    groups feature:
    - 0 -> Disadvantaged group
    - 1 -> Advantaged group

    ``bias_factor`` varies between [0, +inf[, where 0 is completely unbiased.
    """
    rng = np.random.default_rng(random_state)
    groups = pd.Series(rng.binomial(1, 0.5, n_agents), name="groups")
    counts = Counter(groups)
    continuous_cols = [f"f{i}" for i in range(n_continuous)]

    # Generate the input dataset
    X_0 = pd.DataFrame(
        rng.normal(loc=mean, scale=std, size=(counts[0], n_continuous)),
        index=groups[groups == 0].index,
        columns=continuous_cols,
    )

    X_1 = pd.DataFrame(
        rng.normal(
            loc=mean + bias_factor * std, scale=std, size=(counts[1], n_continuous)
        ),
        index=groups[groups == 1].index,
        columns=continuous_cols,
    )

    X = pd.concat([X_0, X_1]).sort_index()

    # TEST: scale continuous features
    if scaler is not None:
        X.loc[:, :] = scaler.transform(X)

    X = pd.concat([X, groups], axis=1)
    X = np.clip(X, 0, 1)

    # Generate the target
    # p0 = 1 / (2 + 2 * bias_factor)
    # p1 = 1 - p0

    # y0 = rng.binomial(1, p0, counts[0])
    # y1 = rng.binomial(1, p1, counts[1])

    # y = pd.concat(
    #     [
    #         pd.Series((y0 if val == 0 else y1), index=group.index)
    #         for val, group in X.groupby("groups")
    #     ]
    # ).sort_index()

    return X  # , y


def get_scaler_hc(
    n_agents=10_000, bias_factor=0, mean=0, std=0.2, random_state=None, **kwargs
):
    X = biased_data_generator_hc(
        n_agents,
        bias_factor=bias_factor,
        mean=mean,
        std=std,
        scaler=None,
        random_state=random_state,
        **kwargs,
    )
    return MinMaxScaler().fit(X[["f0", "f1"]])


def biased_data_generator_hc(
    n_agents, bias_factor=0, mean=0, std=0.2, scaler=None, random_state=None, **kwargs
):
    if "N_LOANS" in kwargs.keys():
        N_LOANS = kwargs["N_LOANS"]

    if "N_AGENTS" in kwargs.keys():
        N_AGENTS = kwargs["N_AGENTS"]

    rng = np.random.default_rng(random_state)

    # For advantaged group
    mu, sigma = mean + 1.5, std
    mu2, sigma2 = mean, std
    high_perf = int((N_LOANS / (2 * N_AGENTS)) * n_agents)
    X1 = rng.normal(mu, sigma, high_perf)
    X2 = rng.normal(mu2, sigma2, int((n_agents / 2) - high_perf))
    f0 = np.concatenate([X1, X2])
    f1 = rng.normal(mean, std, int(n_agents / 2))
    X_adv = np.stack([f0, f1, np.ones(int(n_agents / 2))], axis=1)

    # For disadvantaged group
    mu, sigma = mean + 1.5, std
    mu2, sigma2 = mean - (bias_factor * std), std
    high_perf = int((N_LOANS / (2 * N_AGENTS)) * n_agents)
    X1 = rng.normal(mu, sigma, high_perf)
    X2 = rng.normal(mu2, sigma2, int((n_agents / 2) - high_perf))
    f0 = np.concatenate([X1, X2])
    f1 = rng.normal(mean, std, int(n_agents / 2))
    X_disadv = np.stack([f0, f1, np.zeros(int(n_agents / 2))], axis=1)

    X = pd.DataFrame(np.concatenate([X_adv, X_disadv]), columns=["f0", "f1", "groups"])

    # TEST: scale continuous features
    if scaler is not None:
        X.loc[:, ["f0", "f1"]] = scaler.transform(X[["f0", "f1"]])

    # X = np.clip(X, 0, 1)
    return X
