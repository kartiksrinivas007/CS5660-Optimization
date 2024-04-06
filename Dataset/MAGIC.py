from ucimlrepo import fetch_ucirepo
from scipy.stats import bernoulli
import pandas as pd

def magic_dataset():
    # fetch dataset
    magic_gamma_telescope = fetch_ucirepo(id=159)

    # data (as pandas dataframes)
    X = magic_gamma_telescope.data.features
    y = magic_gamma_telescope.data.targets

    # feature names
    return X, y

def magic04s():
    p = 0.05 # 5% of the features are non-zero
    X, y = magic_dataset()

    # Adding 1000 sparse random features to X
    num_samples = X.shape[0]
    sparse_features = bernoulli.rvs(p, size=[num_samples, 1000])
    df = pd.DataFrame(sparse_features, columns=[f'random_{i}' for i in range(1000)])
    df = df.astype(pd.SparseDtype("float", 0))

    X = pd.concat([X, df], axis=1)
    return X, y

def magic04d():
    p = 0.5 # features take values -1 or 1 with probability 0.5
    X, y = magic_dataset()

    # Adding 1000 dense random features to X
    num_samples = X.shape[0]
    dense_features = 2*bernoulli.rvs(p, size=[num_samples, 1000])-1
    df = pd.DataFrame(dense_features, columns=[f'random_{i}' for i in range(1000)])

    X = pd.concat([X, df], axis=1)
    return X, y


def test_magic04s():
    X, y = magic04s()
    assert X.shape[1] == 1010
    assert X.shape[0] == 19020
    assert y.shape[0] == 19020

def test_magic04d():
    X, y = magic04d()
    assert X.shape[1] == 1010
    assert X.shape[0] == 19020
    assert y.shape[0] == 19020
