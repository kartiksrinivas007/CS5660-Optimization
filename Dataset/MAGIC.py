from ucimlrepo import fetch_ucirepo
from scipy.stats import bernoulli
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split



def prepare_data(X, y):
    X = X.to_numpy().astype('float32')
    y = y.to_numpy().reshape(-1)
    return X, y

def load_cached_into(path_source, path_target, X=None, y=None):
    if not os.path.exists(path_source) or not os.path.exists(path_target):
        return X, y
    else:
        X = pd.read_parquet(path_source)
        y = pd.read_parquet(path_target)
    return X, y


def magic_dataset():
    # fetch dataset
    X = None
    y = None
    
    X, y = load_cached_into('./Dataset/MAGIC_source.gzip', './Dataset/MAGIC_target.gzip', X, y)
    if X is not None and y is not None:
        print("Loading MAGIC from cache...")
        return X, y
    else:     
        magic_gamma_telescope = fetch_ucirepo(id=159)
        # data (as pandas dataframes)
        X = magic_gamma_telescope.data.features
        y = magic_gamma_telescope.data.targets
        X.to_parquet('./Dataset/MAGIC_source.gzip')
        y.to_parquet('./Dataset/MAGIC_target.gzip')
        return X, y

def magic04s():
    X = None
    y = None
    X, y = load_cached_into('./Dataset/MAGIC04s_source.gzip', './Dataset/MAGIC04s_target.gzip', X, y)
    # breakpoint()
    if X is not None and y is not None:
        print("Loading MAGIC04s from cache...")
        return prepare_data(X, y)
    else:
        p = 0.05 # 5% of the features are non-zero
        print("Getting MAGIC...")
        X, y = magic_dataset()
        print("Building MAGIC04s...")
        # Adding 1000 sparse random features to X
        num_samples = X.shape[0]
        sparse_features = bernoulli.rvs(p, size=[num_samples, 1000])
        df = pd.DataFrame(sparse_features, columns=[f'random_{i}' for i in range(1000)])
        # df = df.astype(pd.SparseDtype("float", 0)) # pyarrow cannot dump is sparsity is present
        
        X = pd.concat([X, df], axis=1)
        y = (y == 'h').astype(int)
        
        X.to_parquet('./Dataset/MAGIC04s_source.gzip')
        y.to_parquet('./Dataset/MAGIC04s_target.gzip')
        
        
        return prepare_data(X, y)

def magic04d():
    p = 0.5 # features take values -1 or 1 with probability 0.5
    X, y = magic_dataset()

    # Adding 1000 dense random features to X
    num_samples = X.shape[0]
    dense_features = 2*bernoulli.rvs(p, size=[num_samples, 1000])-1
    df = pd.DataFrame(dense_features, columns=[f'random_{i}' for i in range(1000)])

    X = pd.concat([X, df], axis=1)
    # y contians 'g' and 'h' for 0 and 1 respectively, need to convert to 0 and 1
    y = (y == 'h').astype(int)
    X_numpy = X.to_numpy()
    
    return (X_numpy), y


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


# make the dataset only after normalizing it 

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    # print(data_mean, data_std)
    return X

# need to split the data into train, test and validation sets

def split_data(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.val, random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val

if __name__ == "__main__":
    test_magic04s()
    pass