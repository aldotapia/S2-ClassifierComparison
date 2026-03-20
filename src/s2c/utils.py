#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtw.py

Description: This module implements the Dynamic Time Warping (DTW) algorithm.

Author: Aldo Tapia
Date: 2024-10-18
"""

import re
import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split

def sample_or_boostrap(x: list, limit: int, random_state: int = 1) -> list:
    """
    This function samples or bootstraps a list to a specified limit.

    Parameters
    ----------
    x : list
        The list to be sampled or bootstrapped.
    limit : int
        The limit of the sample or bootstrap.
    random_state : int


    Returns
    -------
    list
        The sampled or bootstrapped DataFrame.
    """
    np.random.seed(random_state)
    if len(x) > limit:
        res = np.random.choice(x, limit, replace=False)
    else:
        res = np.random.choice(x, len(x), replace=False)
        res2 = np.random.choice(x, size=limit-len(x), replace=True)
        res = np.concatenate([res, res2])
    return res



def kfold(idx: list, k: int, shuffle: bool = False, random_state: int = 42) -> list:
    """
    Split the data into ordered k-folds
    for cross-validation of time series data

    Parameters
    ----------
        idx: list of indices
        k: number of folds
        shuffle: whether to shuffle the data before splitting
        random_state: random seed for shuffling

    Returns
    -------
        list of tuples of train and test indices

    """
    np.random.seed(random_state)
    n = len(idx)
    fold_size = n//k
    folds = []
    if shuffle:
        idx = np.random.choice(idx, len(idx), replace=False)
    for i in range(k):
        test = idx[i*fold_size:(i+1)*fold_size]
        train = [x for x in idx if x not in test]
        folds.append((train, test))
    return folds


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps in the time series data using linear interpolation
    and forward and backward filling

    Parameters
    ----------
        df: pandas dataframe

    Returns
    -------
        pandas dataframe with filled gaps

    """
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    dates = sorted(df['date'].unique())
    ids = sorted(df['id'].unique())
    bands = ['B02', 'B03', 'B04', 'B08']
    
    template = pd.DataFrame({'date': dates})
    
    dfs = []
    
    for id in ids:
        l = []
        di = {}
        for band in bands:
            df_temp = pd.merge(template, df[(df['id']==id) & (df['band']==band)], how='left', on=['date'])
            df_temp['rho'] = df_temp['rho'].interpolate(method='linear')
            df_temp['rho'] = df_temp['rho'].ffill()
            df_temp['rho'] = df_temp['rho'].bfill()
            df_temp = df_temp[(df_temp['date'] >= '2021-05-01') & (df_temp['date'] <= '2022-04-30')]
            if band == 'B02':
                di['date'] = df_temp['date'].dt.strftime('%Y-%m-%d').tolist()
                di['id'] = [id]*len(di['date'])
            di[f'{band}'] = df_temp['rho'].values
        dfs.append(pd.DataFrame(di))
    
    return pd.concat(dfs)


def to_integer(dt_time: pd.Timestamp,
               initial: pd.Timestamp = pd.Timestamp('1970-01-01')) -> int:
    """
    Convert a datetime object to an integer

    Parameters
    ----------
        dt_time: pandas datetime object
        initial: initial date
        
    Returns
    -------
        integer
    """
    return (dt_time - initial).dt.days

def prepare_data(
        df: pd.DataFrame,
        label2int: dict,
        n_classes: int = 16,
        dim: int = 1,
        experiment: int = 1,
        split: float = None,
        shuffle: bool = False,
        seed: int = 42,
        transpose: bool = False,
        normalize: bool = False,
        norm_path: str = '',
        load_norm: bool = False
        ):
    """
    Prepare data for training a neural network

    Parameters
    ----------
    df: pandas dataframe
    label2int: dictionary mapping class labels to integers
    n_classes: number of classes
    dim: dimension of the input data (1 or 2)
    experiment: experiment number
    split: proportion of data to use for validation
    shuffle: whether to shuffle the data before splitting
    transpose: whether to transpose the input data
    normalize: whether to normalize the input data
    norm_path: path to save normalization values
    load_norm: whether to load normalization values from norm_path
    
    Returns
    -------
    X: numpy array of input data
    y: numpy array of target data
    X_val: numpy array of validation input data
    y_val: numpy array of validation target data
    """

    if experiment >= 2:
        df['ndvi'] = (df['B08'] - df['B04']) / (df['B08'] + df['B04'] + 0.000001)
        df['grvi'] = (df['B03'] - df['B04']) / (df['B03'] + df['B04'] + 0.000001)
        df['gndvi'] = (df['B08'] - df['B03']) / (df['B08'] + df['B03'] + 0.000001)
        df['evi'] = 2.5 * (df['B08'] - df['B04']) / (df['B08'] + 6 * df['B04'] - 7.5 * df['B02'] + 1)
        #df = df.replace([np.inf, -np.inf], np.nan)
        #df = df.fillna(0)

    if normalize:
        
        if load_norm:
            try:
                norm_vals = pd.read_csv(norm_path)
                for _, row in norm_vals.iterrows():
                    band = row['band']
                    mean_ = row['mean']
                    std_ = row['std']
                    df[band] = (df[band] - mean_) / std_
            except Exception as e:
                print(f"Error loading normalization values: {e}")
                print("Proceeding without normalization.")
        else:
            l = []
            if experiment == 1:
                for band in ['B02', 'B03', 'B04', 'B08']:
                    mean_ = df[band].mean()
                    std_ = df[band].std()
                    df[band] = (df[band] - mean_) / std_
                    l.append({'band': band, 'mean': mean_, 'std': std_})
            if experiment >= 2:
                for band in ['ndvi', 'grvi', 'gndvi', 'evi']:
                    mean_ = df[band].mean()
                    std_ = df[band].std()
                    df[band] = (df[band] - mean_) / std_
                    l.append({'band': band, 'mean': mean_, 'std': std_})
            norm_vals = pd.DataFrame(l)
            if norm_path != '':
                norm_vals.to_csv(norm_path, index=False)

    ids = df['id'].unique()
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(ids)
    # split should be a float between 0 and 1
    if split is not None:
        if split < 0 or split > 1:
            raise ValueError("split must be a float between 0 and 1")
        ids = train_test_split(ids, test_size=split, random_state=42)

        X_list = []
        y_list = []
        for id in ids[0]:
            temp = df[df['id'] == id]
            if dim == 1:
                if experiment == 1:
                    Xtemp = np.concatenate([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy()], axis=0)
                if experiment == 2:
                    Xtemp = np.concatenate([temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()], axis=0)
                if experiment == 3:
                    Xtemp = np.concatenate([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy(),
                                            temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()], axis=0)
            elif dim == 2:
                if experiment == 1:
                    Xtemp = np.vstack([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy()])
                if experiment == 2:
                    Xtemp = np.vstack([temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()])
                if experiment == 3:
                    Xtemp = np.vstack([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy(),
                                        temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()])
                Xtemp = Xtemp.reshape(1, Xtemp.shape[0], Xtemp.shape[1])
            else:
                raise ValueError("dim must be 1 or 2")
            ytemp = label2int[temp['class'].iloc[0]]
            X_list.append(Xtemp)
            ytemp = torch.nn.functional.one_hot(torch.tensor(ytemp), n_classes).numpy()
            y_list.append(ytemp)
            
        X = np.vstack(X_list)
        y = np.vstack(y_list)

        X_list_val = []
        y_list_val = []
        for id in ids[1]:
            temp = df[df['id'] == id]
            if dim == 1:
                if experiment == 1:
                    Xtemp = np.concatenate([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy()], axis=0)
                if experiment == 2:
                    Xtemp = np.concatenate([temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()], axis=0)
                if experiment == 3:
                    Xtemp = np.concatenate([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy(),
                                            temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()], axis=0)
            elif dim == 2:
                if experiment == 1:
                    Xtemp = np.vstack([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy()])
                if experiment == 2:
                    Xtemp = np.vstack([temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()])
                if experiment == 3:
                    Xtemp = np.vstack([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy(),
                                        temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()])
                Xtemp = Xtemp.reshape(1, Xtemp.shape[0], Xtemp.shape[1])
            else:
                raise ValueError("dim must be 1 or 2")             
            ytemp = label2int[temp['class'].iloc[0]]
            X_list_val.append(Xtemp)
            ytemp = torch.nn.functional.one_hot(torch.tensor(ytemp), n_classes).numpy()
            y_list_val.append(ytemp)

        X_val = np.vstack(X_list_val)
        y_val = np.vstack(y_list_val)

    else:
        X_list = []
        y_list = []
        for id in ids:
            temp = df[df['id'] == id]
            if dim == 1:
                if experiment == 1:
                    Xtemp = np.concatenate([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy()], axis=0)
                if experiment == 2:
                    Xtemp = np.concatenate([temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()], axis=0)
                if experiment == 3:
                    Xtemp = np.concatenate([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy(),
                                            temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()], axis=0)
            elif dim == 2:
                if experiment == 1:
                    Xtemp = np.vstack([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy()])
                if experiment == 2:
                    Xtemp = np.vstack([temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()])
                if experiment == 3:
                    Xtemp = np.vstack([temp['B02'].to_numpy(), temp['B03'].to_numpy(), temp['B04'].to_numpy(), temp['B08'].to_numpy(),
                                        temp['ndvi'].to_numpy(), temp['grvi'].to_numpy(), temp['gndvi'].to_numpy(), temp['evi'].to_numpy()])
                Xtemp = Xtemp.reshape(1, Xtemp.shape[0], Xtemp.shape[1])
            else:
                raise ValueError("dim must be 1 or 2")
            ytemp = label2int[temp['class'].iloc[0]]
            X_list.append(Xtemp)
            ytemp = torch.nn.functional.one_hot(torch.tensor(ytemp), n_classes).numpy()
            y_list.append(ytemp)
        
        X = np.vstack(X_list)
        y = np.vstack(y_list)
        X_val = ids
        y_val = None
    if transpose:
        X = np.transpose(X, (0, 2, 1))
        if split is not None:
            X_val = np.transpose(X_val, (0, 2, 1))
    return X, y, X_val, y_val

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def read_results(path: str) -> pd.DataFrame:
    """
    Read results from a file and return a DataFrame

    Parameters
    ----------
    path: str
        Path to the results file
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the results
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    rows = []
    
    for line in lines:
        row = {}
        parts = line.strip().split(",")
    
        for part in parts:
            part = part.strip()
            if " " not in part:
                continue
    
            key, val = part.split(" ", 1)
            val = val.strip()
    
            # Convert value to correct Python type
            if val == "None":
                val = "None"
            else:
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    # string like "sqrt"
                    pass
    
            row[key] = val
    
        rows.append(row)
    
    return pd.DataFrame(rows)

