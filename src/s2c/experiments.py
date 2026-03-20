#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtw.py

Description: This module implements the Dynamic Time Warping (DTW) algorithm.

Author: Aldo Tapia
Date: 2024-11-21
"""

import time
import pandas as pd
from dask import delayed, compute
from enum import Enum
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

try:
    from utils import *
    from dtw import dtw
    from ann import *
except ImportError:
    from .utils import *
    from .dtw import dtw
    from .ann import *


class Experiment(Enum):
    KIND1 = 1
    KIND2 = 2
    KIND3 = 3

def dtw_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str
        ) -> None:
    """
    Function to run DTW experiments based on the splits provided.

    Parameters
    ----------
    db_path : str
        Path to the database.
    splits_path : str
        Path to the splits.
    output_path : str
        Path to save the results.
    experiment : int
        Kin of experiment to run (1, 2, 3):
            1: Bands
            2: Spectral indices
            3: Bands + Spectral indices

    Returns
    -------
    None
    """

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)
    #db['date'] = to_integer(pd.to_datetime(db['date']))

    if experiment.value >= 2:
        db['ndvi'] = (db['B08'] - db['B04']) / (db['B08'] + db['B04'] + 1e-6)
        db['grvi'] = (db['B03'] - db['B04']) / (db['B03'] + db['B04'] + 1e-6)
        db['gndvi'] = (db['B08'] - db['B03']) / (db['B08'] + db['B03'] + 1e-6)
        db['evi'] = 2.5 * (db['B08'] - db['B04']) / (db['B08'] + 6 * db['B04'] - 7.5 * db['B02'] + 1)
    
    if experiment.value == 1:
        channels = ['B02', 'B03', 'B04', 'B08']
    
    elif experiment.value == 2:
        channels = ['ndvi', 'grvi', 'gndvi', 'evi']

    elif experiment.value == 3:
        channels = ['B02', 'B03', 'B04', 'B08', 'ndvi', 'grvi', 'gndvi', 'evi']
    
    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):

        t1 = time.time()

        results = [
            {
                'fold': fold,
                'id_sample': sample_id,
                'id_pattern': pattern_id,
                'dist': delayed(dtw)(
                    x=db[db['id'] == sample_id][channels].to_numpy(),
                    y=db[db['id'] == pattern_id][channels].to_numpy(),
                    dissimilarity=0
                ),
                'class1': db[db['id'] == sample_id]['class'].values[0],
                'class2': db[db['id'] == pattern_id]['class'].values[0]
            }
            for sample in [dfsplitted[(dfsplitted['split'] == 'test') & (dfsplitted['fold'] == fold)]]
            for pattern in [dfsplitted[(dfsplitted['split'] == 'train') & (dfsplitted['fold'] == fold)]]
            for sample_id in sample['id']
            for pattern_id in pattern['id']
        ]
        
        results = compute(results, scheduler='processes')

        t2 = time.time()
        print(f'Time: {t2-t1}')

        name = f'dtw_exp{experiment.value}_fold{fold}.csv'

        pd.DataFrame(results[0]).to_csv(f'{output_path}/{name}', index=False)


def twdtw_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        alpha: float,
        beta: int
        ) -> None:
    """
    Function to run TWDTW experiments based on the splits provided.

    Parameters
    ----------
    db_path : str
        Path to the database.
    splits_path : str
        Path to the splits.
    output_path : str
        Path to save the results.
    experiment : int
        Kin of experiment to run (1, 2, 3):
            1: Bands
            2: Spectral indices
            3: Bands + Spectral indices
    alpha : float
        Alpha parameter for the TWDTW algorithm.
    beta : int
        Beta parameter for the TWDTW algorithm.

    Returns
    -------
    None
    """

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)
    db['date'] = to_integer(pd.to_datetime(db['date']))

    if experiment.value >= 2:
        db['ndvi'] = (db['B08'] - db['B04']) / (db['B08'] + db['B04'])
        db['grvi'] = (db['B03'] - db['B04']) / (db['B03'] + db['B04'])
        db['gndvi'] = (db['B08'] - db['B03']) / (db['B08'] + db['B03'])
        db['evi'] = 2.5 * (db['B08'] - db['B04']) / (db['B08'] + 6 * db['B04'] - 7.5 * db['B02'] + 1)
    
    if experiment.value == 1:
        channels = ['B02', 'B03', 'B04', 'B08']
    
    elif experiment.value == 2:
        channels = ['ndvi', 'grvi', 'gndvi', 'evi']

    elif experiment.value == 3:
        channels = ['B02', 'B03', 'B04', 'B08', 'ndvi', 'grvi', 'gndvi', 'evi']
    
    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):

        t1 = time.time()

        results = [
            {
                'fold': fold,
                'id_sample': sample_id,
                'id_pattern': pattern_id,
                'dist': delayed(dtw)(
                    x=db[db['id'] == sample_id][channels].to_numpy(),
                    y=db[db['id'] == pattern_id][channels].to_numpy(),
                    dissimilarity=0,
                    tx=db[db['id'] == sample_id]['date'].to_numpy(),
                    ty=db[db['id'] == pattern_id]['date'].to_numpy(),
                    alpha=alpha,
                    beta=beta

                ),
                'class1': db[db['id'] == sample_id]['class'].values[0],
                'class2': db[db['id'] == pattern_id]['class'].values[0]
            }
            for sample in [dfsplitted[(dfsplitted['split'] == 'test') & (dfsplitted['fold'] == fold)]]
            for pattern in [dfsplitted[(dfsplitted['split'] == 'train') & (dfsplitted['fold'] == fold)]]
            for sample_id in sample['id']
            for pattern_id in pattern['id']
        ]
        
        results = compute(results, scheduler='processes')

        t2 = time.time()
        print(f'Time: {t2-t1}')

        name = f'twdtw_exp{experiment.value}_{str(alpha).replace("0.","")}_{beta}_fold{fold}.csv'

        pd.DataFrame(results[0]).to_csv(f'{output_path}/{name}', index=False)

def ann_a_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        l1: int = 512,
        l2: int = 256,
        l3: int = 128,
        dropout: float = 0.3,
        epochs: int = 200,
        lr = 0.0001,
        es_patience = 30
        ) -> None:
    """
    Function to run ANN experiments based on the splits provided.

    Parameters
    ----------
    db_path : str
        Path to the database.
    splits_path : str
        Path to the splits.
    output_path : str
        Path to save the results.
    experiment : int
        Kin of experiment to run (1, 2, 3):
            1: Bands
            2: Spectral indices
            3: Bands + Spectral indices
    input_dim : int
        Dimension of the input data.
    output_dim : int
        Dimension of the output data.
    l1 : int
        Number of neurons in the first hidden layer.
    l2 : int
        Number of neurons in the second hidden layer.
    l3 : int
        Number of neurons in the third hidden layer.
    dropout : float
        Dropout rate for regularization.
    epochs : int
        Number of epochs for training.
    lr : float
        Learning rate for the optimizer.
    es_patience : int
        Patience for early stopping.

    Returns
    -------
    None
    """

    set_seed(43)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):

        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        X_train, y_train, X_val, y_val = prepare_data(db_train, label2int, split=0.2, dim=1, experiment=experiment.value, shuffle=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_A_Exp{experiment.value}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.csv')
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=1, experiment=experiment.value, shuffle=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_A_Exp{experiment.value}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.csv', load_norm=True)

        criterion = nn.CrossEntropyLoss()

        model = NeutalNetA(
            input_dim=X_train.shape[1],
            output_dim=len(labels),
            l1 = l1,
            l2 = l2,
            l3 = l3,
            dropout = dropout
            ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).squeeze())
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).squeeze())
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        early_stopper = EarlyStopper(patience=es_patience, min_delta=0.001)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        patience_lr = 10     # No improvement for X epochs -> adjust LR
        factor = 0.5              # Factor to multiply/divide LR by
        min_lr = 0.00001
        
        history = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, torch.argmax(y, dim=1))
                total_loss.append(loss.item())
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = np.mean(total_loss)
        
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                # Compute validation loss
                yval_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_loss = criterion(yval_pred, torch.argmax(torch.tensor(y_val, dtype=torch.float32).to(device), dim=1))

                # Compute accuracy
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.argmax(dim=1)).sum().item()
                print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {100 * correct / total}")
        
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_loss': val_loss.item(),
                'accuracy': 100 * correct / total
            })
        
            # Save best model and adjust learning rate
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f'{output_path}/ANN_A_model_exp{experiment.value}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.pth')
                print(f"Model saved at epoch {epoch}")
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
                # Reduce learning rate if no improvement
                if epochs_no_improve >= patience_lr:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = max(old_lr * factor, min_lr)
                        param_group['lr'] = new_lr
                        print(f"Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    
                    epochs_no_improve = 0
                
            # Early stopping check
            if early_stopper.early_stop(val_loss):             
                print(f"Early stopping at epoch {epoch}")
                break
        
        model.load_state_dict(torch.load(f'{output_path}/ANN_A_model_exp{experiment.value}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.pth'))

        df = pd.DataFrame(history)

        # reset plot
        plt.clf()
        plt.plot(df['epoch'], df['loss'], label='loss')
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(f'{output_path}/ANN_A_history_exp{experiment.value}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.png')

        # predict
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).squeeze())
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                print(f'Batch {i+1}/{len(test_loader)}')
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())


        y_pred = np.vstack(y_pred_list)
        y_true = np.vstack(y_true_list)
        label_pred = np.argmax(y_pred, axis=1)
        label_true = np.argmax(y_true, axis=1)
        acc = (label_pred == label_true).sum() / len(label_true)
        print(f"Accuracy: {acc}")
        
        # saving accuracy
        with open(f'{output_path}/ANN_A_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, l1 {l1}, l2 {l2}, l3 {l3}, dropout {dropout}, lr {lr}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(y_pred, columns=labels)
        df_pred['label_pred'] = label_pred
        df_pred['label_true'] = label_true
        df_pred['id_sample'] = ids
        
        df_pred.to_csv(f'{output_path}/ANN_A_predictions_exp{experiment.value}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.csv', index=False)

        t2 = time.time()
        print(f'Time: {t2-t1}')


def ann_b_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        conv1_channels: int = 64,
        conv2_channels: int = 32,
        dense1: int = 256,
        dense2: int = 128,
        dense3: int = 64,
        dropout: float = 0.3,
        epochs: int = 200,
        lr = 0.0001,
        es_patience = 30
        ) -> None:
    """
    Function to run 1D CNN experiments based on the splits provided.
    """

    set_seed(43)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):
        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        # Prepare data for 1D CNN - note we use dim=2 to get the sequence dimension
        X_train, y_train, X_val, y_val = prepare_data(db_train, label2int, split=0.2, dim=2, experiment=experiment.value, shuffle=True, transpose=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_B_Exp{experiment.value}_{conv1_channels}_{conv2_channels}_{dense1}_{dense2}_{dense3}_{dropout}_{lr}_fold{fold}.csv')
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=2, experiment=experiment.value, shuffle=True, normalize=True, transpose=True, norm_path=f'{output_path}/norm_vals_ANN_B_Exp{experiment.value}_{conv1_channels}_{conv2_channels}_{dense1}_{dense2}_{dense3}_{dropout}_{lr}_fold{fold}.csv', load_norm=True)

        # Initialize model with proper input dimensions for 1D CNN
        input_channels = X_train.shape[1]  # Number of features
        sequence_length = X_train.shape[2]  # Sequence length

        criterion = nn.CrossEntropyLoss()

        model = NeuralNetB(
            input_channels=input_channels,
            sequence_length=sequence_length,
            output_dim=len(labels),
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            dense1=dense1,
            dense2=dense2,
            dense3=dense3,
            dropout=dropout
        ).to(device)

        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).squeeze()
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).squeeze()
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        early_stopper = EarlyStopper(patience=es_patience, min_delta=0.001)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        patience_lr = 10     # No improvement for X epochs -> adjust LR
        factor = 0.5              # Factor to multiply/divide LR by
        min_lr = 0.00001
        
        history = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, torch.argmax(y, dim=1))
                total_loss.append(loss.item())
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = np.mean(total_loss)
        
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                # Compute validation loss
                yval_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_loss = criterion(yval_pred, torch.argmax(torch.tensor(y_val, dtype=torch.float32).to(device), dim=1))

                # Compute accuracy
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.argmax(dim=1)).sum().item()
                print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {100 * correct / total}")
        
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_loss': val_loss.item(),
                'accuracy': 100 * correct / total
            })

            # Save best model and adjust learning rate
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f'{output_path}/ANN_B_model_exp{experiment.value}_{conv1_channels}_{conv2_channels}_{dense1}_{dense2}_{dense3}_{dropout}_{lr}_fold{fold}.pth')
                print(f"Model saved at epoch {epoch}")
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
                # Reduce learning rate if no improvement
                if epochs_no_improve >= patience_lr:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = max(old_lr * factor, min_lr)
                        param_group['lr'] = new_lr
                        print(f"Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    
                    epochs_no_improve = 0
                
            # Early stopping check
            if early_stopper.early_stop(val_loss):             
                print(f"Early stopping at epoch {epoch}")
                break
        

        # Save and evaluate model
        model.load_state_dict(torch.load(f'{output_path}/ANN_B_model_exp{experiment.value}_{conv1_channels}_{conv2_channels}_{dense1}_{dense2}_{dense3}_{dropout}_{lr}_fold{fold}.pth'))
        
        df = pd.DataFrame(history)
        
        # Plot training history
        plt.clf()
        plt.plot(df['epoch'], df['loss'], label='loss')
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(f'{output_path}/ANN_B_history_exp{experiment.value}_{conv1_channels}_{conv2_channels}_{dense1}_{dense2}_{dense3}_{dropout}_{lr}_fold{fold}.png')

        # Final predictions
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).squeeze()
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())

        # Save predictions
        y_pred = np.vstack(y_pred_list)
        y_true = np.vstack(y_true_list)
        label_pred = np.argmax(y_pred, axis=1)
        label_true = np.argmax(y_true, axis=1)
        acc = (label_pred == label_true).sum() / len(label_true)
        print(f"Accuracy: {acc}")

        # saving accuracy
        with open(f'{output_path}/ANN_B_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, conv1_channels {conv1_channels}, conv2_channels {conv2_channels}, dense1 {dense1}, dense2 {dense2}, dense3 {dense3}, dropout {dropout}, lr {lr}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(y_pred, columns=labels)
        df_pred['label_pred'] = label_pred
        df_pred['label_true'] = label_true
        df_pred['id_sample'] = ids

        df_pred.to_csv(f'{output_path}/ANN_B_predictions_exp{experiment.value}_{conv1_channels}_{conv2_channels}_{dense1}_{dense2}_{dense3}_{dropout}_{lr}_fold{fold}.csv', index=False)

        t2 = time.time()
        print(f'Time: {t2-t1}')



def ann_c_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        n_components: int = 15,
        l1: int = 512,
        l2: int = 256,
        l3: int = 128,
        dropout: float = 0.3,
        epochs: int = 200,
        lr = 0.0001,
        es_patience = 30
        ) -> None:
    """
    Function to run ANN experiments based on the splits provided.

    Parameters
    ----------
    db_path : str
        Path to the database.
    splits_path : str
        Path to the splits.
    output_path : str
        Path to save the results.
    experiment : int
        Kin of experiment to run (1, 2, 3):
            1: Bands
            2: Spectral indices
            3: Bands + Spectral indices
    input_dim : int
        Dimension of the input data.
    output_dim : int
        Dimension of the output data.
    n_components : int
        Number of components for PCA.
    l1 : int
        Number of neurons in the first hidden layer.
    l2 : int
        Number of neurons in the second hidden layer.
    l3 : int
        Number of neurons in the third hidden layer.
    epochs : int
        Number of epochs for training.
    lr : float
        Learning rate for the optimizer.
    momentum : float
        Momentum for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    patience : int
        Patience for the early stopping.

    Returns
    -------
    None
    """

    set_seed(43)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):

        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        X_train, y_train, X_val, y_val = prepare_data(db_train, label2int, split=0.2, dim=1, experiment=experiment.value, shuffle=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_C_Exp{experiment.value}_pca{n_components}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.csv')
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=1, experiment=experiment.value, shuffle=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_C_Exp{experiment.value}_pca{n_components}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.csv', load_norm=True)

        pca = PCA(n_components=n_components)

        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

        criterion = nn.CrossEntropyLoss()

        model = NeutalNetA(
            input_dim=X_train.shape[1],
            output_dim=len(labels),
            l1 = l1,
            l2 = l2,
            l3 = l3,
            dropout = dropout
            ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).squeeze())
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).squeeze())
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        early_stopper = EarlyStopper(patience=es_patience, min_delta=0.001)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        patience_lr = 10     # No improvement for X epochs -> adjust LR
        factor = 0.5              # Factor to multiply/divide LR by
        min_lr = 0.00001
        
        history = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, torch.argmax(y, dim=1))
                total_loss.append(loss.item())
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = np.mean(total_loss)
        
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                # Compute validation loss
                yval_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_loss = criterion(yval_pred, torch.argmax(torch.tensor(y_val, dtype=torch.float32).to(device), dim=1))

                # Compute accuracy
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.argmax(dim=1)).sum().item()
                print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {100 * correct / total}")
        
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_loss': val_loss.item(),
                'accuracy': 100 * correct / total
            })
        
            # Save best model and adjust learning rate
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f'{output_path}/ANN_C_model_exp{experiment.value}_pca{n_components}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.pth')
                print(f"Model saved at epoch {epoch}")
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
                # Reduce learning rate if no improvement
                if epochs_no_improve >= patience_lr:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = max(old_lr * factor, min_lr)
                        param_group['lr'] = new_lr
                        print(f"Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    
                    epochs_no_improve = 0
                
            # Early stopping check
            if early_stopper.early_stop(val_loss):             
                print(f"Early stopping at epoch {epoch}")
                break
        
        model.load_state_dict(torch.load(f'{output_path}/ANN_C_model_exp{experiment.value}_pca{n_components}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.pth'))

        df = pd.DataFrame(history)

        # reset plot
        plt.clf()
        plt.plot(df['epoch'], df['loss'], label='loss')
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(f'{output_path}/ANN_C_history_exp{experiment.value}_pca{n_components}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.png')

        # predict
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).squeeze())
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                print(f'Batch {i+1}/{len(test_loader)}')
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())


        y_pred = np.vstack(y_pred_list)
        y_true = np.vstack(y_true_list)
        label_pred = np.argmax(y_pred, axis=1)
        label_true = np.argmax(y_true, axis=1)
        acc = (label_pred == label_true).sum() / len(label_true)
        print(f"Accuracy: {acc}")

        # saving accuracy
        with open(f'{output_path}/ANN_C_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, l1 {l1}, l2 {l2}, l3 {l3}, dropout {dropout}, lr {lr}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(y_pred, columns=labels)
        df_pred['label_pred'] = label_pred
        df_pred['label_true'] = label_true
        df_pred['id_sample'] = ids
        
        df_pred.to_csv(f'{output_path}/ANN_C_predictions_exp{experiment.value}_pca{n_components}_{l1}_{l2}_{l3}_{dropout}_{lr}_fold{fold}.csv', index=False)

        t2 = time.time()
        print(f'Time: {t2-t1}')


def ann_d_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        conv1_channels: int = 64,
        conv2_channels: int = 32,
        conv3_channels: int = 16,
        dense1: int = 256,
        dense2: int = 128,
        dropout: float = 0.3,
        epochs: int = 200,
        lr = 0.0001,
        es_patience = 30
        ) -> None:
    """
    Function to run 1D CNN experiments based on the splits provided.
    """

    set_seed(43)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):
        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        # Prepare data for 1D CNN - note we use dim=2 to get the sequence dimension
        X_train, y_train, X_val, y_val = prepare_data(db_train, label2int, split=0.2, dim=2, experiment=experiment.value, shuffle=True, transpose=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_D_Exp{experiment.value}_{conv1_channels}_{conv2_channels}_{conv3_channels}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv')
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=2, experiment=experiment.value, shuffle=True, normalize=True, transpose=True, norm_path=f'{output_path}/norm_vals_ANN_D_Exp{experiment.value}_{conv1_channels}_{conv2_channels}_{conv3_channels}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv', load_norm=True)

        # Initialize model with proper input dimensions for 1D CNN
        input_channels = X_train.shape[1]  # Number of features
        sequence_length = X_train.shape[2]  # Sequence length

        criterion = nn.CrossEntropyLoss()

        model = NeuralNetC(
            input_channels=input_channels,
            sequence_length=sequence_length,
            output_dim=len(labels),
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            conv3_channels=conv3_channels,
            dense1=dense1,
            dense2=dense2,
            dropout=dropout
        ).to(device)

        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).squeeze()
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).squeeze()
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        early_stopper = EarlyStopper(patience=es_patience, min_delta=0.001)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        patience_lr = 10     # No improvement for X epochs -> adjust LR
        factor = 0.5              # Factor to multiply/divide LR by
        min_lr = 0.00001
        
        history = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, torch.argmax(y, dim=1))
                total_loss.append(loss.item())
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = np.mean(total_loss)
        
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                # Compute validation loss
                yval_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_loss = criterion(yval_pred, torch.argmax(torch.tensor(y_val, dtype=torch.float32).to(device), dim=1))

                # Compute accuracy
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.argmax(dim=1)).sum().item()
                print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {100 * correct / total}")
        
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_loss': val_loss.item(),
                'accuracy': 100 * correct / total
            })

            # Save best model and adjust learning rate
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f'{output_path}/ANN_D_model_exp{experiment.value}_{conv1_channels}_{conv2_channels}_{conv3_channels}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.pth')
                print(f"Model saved at epoch {epoch}")
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
                # Reduce learning rate if no improvement
                if epochs_no_improve >= patience_lr:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = max(old_lr * factor, min_lr)
                        param_group['lr'] = new_lr
                        print(f"Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    
                    epochs_no_improve = 0
                
            # Early stopping check
            if early_stopper.early_stop(val_loss):             
                print(f"Early stopping at epoch {epoch}")
                break
        

        # Save and evaluate model
        model.load_state_dict(torch.load(f'{output_path}/ANN_D_model_exp{experiment.value}_{conv1_channels}_{conv2_channels}_{conv3_channels}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.pth'))
        
        df = pd.DataFrame(history)
        
        # Plot training history
        plt.clf()
        plt.plot(df['epoch'], df['loss'], label='loss')
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(f'{output_path}/ANN_D_history_exp{experiment.value}_{conv1_channels}_{conv2_channels}_{conv3_channels}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.png')

        # Final predictions
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).squeeze()
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())

        # Save predictions
        y_pred = np.vstack(y_pred_list)
        y_true = np.vstack(y_true_list)
        label_pred = np.argmax(y_pred, axis=1)
        label_true = np.argmax(y_true, axis=1)
        acc = (label_pred == label_true).sum() / len(label_true)
        print(f"Accuracy: {acc}")

        # saving accuracy
        with open(f'{output_path}/ANN_D_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, conv1_channels {conv1_channels}, conv2_channels {conv2_channels}, conv3_channels {conv3_channels}, dense1 {dense1}, dense2 {dense2}, dropout {dropout}, lr {lr}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(y_pred, columns=labels)
        df_pred['label_pred'] = label_pred
        df_pred['label_true'] = label_true
        df_pred['id_sample'] = ids

        df_pred.to_csv(f'{output_path}/ANN_D_predictions_exp{experiment.value}_{conv1_channels}_{conv2_channels}_{conv3_channels}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv', index=False)

        t2 = time.time()
        print(f'Time: {t2-t1}')



def ann_e_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        projection_size: int = 128,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        dense1: int = 256,
        dense2: int = 128,
        dropout: float = 0.3,
        epochs: int = 200,
        lr = 0.0001,
        es_patience = 30
        ) -> None:
    """
    Function to run LSTM experiments based on the splits provided.
    """

    set_seed(43)

    device = (
        "cuda:1"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):
        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        # Prepare data for LSTM - note we use dim=2 to get the sequence dimension
        X_train, y_train, X_val, y_val = prepare_data(db_train, label2int, split=0.2, dim=2, experiment=experiment.value, shuffle=True, transpose=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_E_Exp{experiment.value}_{projection_size}_{lstm_hidden_size}_{lstm_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv')
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=2, experiment=experiment.value, shuffle=True, normalize=True, transpose=True, norm_path=f'{output_path}/norm_vals_ANN_E_Exp{experiment.value}_{projection_size}_{lstm_hidden_size}_{lstm_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv', load_norm=True)

        # Initialize model with proper input dimensions for LSTM
        input_channels = X_train.shape[2]  # Number of features
        # sequence_length = X_train.shape[1]  # Sequence length

        criterion = nn.CrossEntropyLoss()

        model = NeuralNetD(
            input_channels=input_channels,
            projection_size=projection_size,
            output_dim=len(labels),
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dense1=dense1,
            dense2=dense2,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).squeeze()
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).squeeze()
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        early_stopper = EarlyStopper(patience=es_patience, min_delta=0.001)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        patience_lr = 10     # No improvement for X epochs -> adjust LR
        factor = 0.5              # Factor to multiply/divide LR by
        min_lr = 0.00001
        
        history = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, torch.argmax(y, dim=1))
                total_loss.append(loss.item())
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = np.mean(total_loss)
        
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                # Compute validation loss
                yval_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_loss = criterion(yval_pred, torch.argmax(torch.tensor(y_val, dtype=torch.float32).to(device), dim=1))

                # Compute accuracy
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.argmax(dim=1)).sum().item()
                print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {100 * correct / total}")
        
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_loss': val_loss.item(),
                'accuracy': 100 * correct / total
            })

            # Save best model and adjust learning rate
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f'{output_path}/ANN_E_model_exp{experiment.value}_{projection_size}_{lstm_hidden_size}_{lstm_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.pth')
                print(f"Model saved at epoch {epoch}")
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
                # Reduce learning rate if no improvement
                if epochs_no_improve >= patience_lr:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = max(old_lr * factor, min_lr)
                        param_group['lr'] = new_lr
                        print(f"Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    
                    epochs_no_improve = 0
                
            # Early stopping check
            if early_stopper.early_stop(val_loss):             
                print(f"Early stopping at epoch {epoch}")
                break
        

        # Save and evaluate model
        model.load_state_dict(torch.load(f'{output_path}/ANN_E_model_exp{experiment.value}_{projection_size}_{lstm_hidden_size}_{lstm_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.pth'))
        
        df = pd.DataFrame(history)
        
        # Plot training history
        plt.clf()
        plt.plot(df['epoch'], df['loss'], label='loss')
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(f'{output_path}/ANN_E_history_exp{experiment.value}_{projection_size}_{lstm_hidden_size}_{lstm_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.png')

        # Final predictions
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).squeeze()
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())

        # Save predictions
        y_pred = np.vstack(y_pred_list)
        y_true = np.vstack(y_true_list)
        label_pred = np.argmax(y_pred, axis=1)
        label_true = np.argmax(y_true, axis=1)
        acc = (label_pred == label_true).sum() / len(label_true)
        print(f"Accuracy: {acc}")

        # saving accuracy
        with open(f'{output_path}/ANN_E_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, projection_size {projection_size}, lstm_hidden_size {lstm_hidden_size}, lstm_num_layers {lstm_num_layers}, dense1 {dense1}, dense2 {dense2}, dropout {dropout}, lr {lr}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(y_pred, columns=labels)
        df_pred['label_pred'] = label_pred
        df_pred['label_true'] = label_true
        df_pred['id_sample'] = ids

        df_pred.to_csv(f'{output_path}/ANN_E_predictions_exp{experiment.value}_{projection_size}_{lstm_hidden_size}_{lstm_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv', index=False)

        t2 = time.time()
        print(f'Time: {t2-t1}')


def ann_f_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        projection_size: int = 128,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        dense1: int = 256,
        dense2: int = 128,
        dropout: float = 0.3,
        epochs: int = 200,
        lr = 0.0001,
        es_patience = 30
        ) -> None:
    """
    Function to run GRU experiments based on the splits provided.
    """

    set_seed(43)

    device = (
        "cuda:2"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):
        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        # Prepare data for GRU - note we use dim=2 to get the sequence dimension
        X_train, y_train, X_val, y_val = prepare_data(db_train, label2int, split=0.2, dim=2, experiment=experiment.value, shuffle=True, transpose=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_F_Exp{experiment.value}_{projection_size}_{gru_hidden_size}_{gru_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv')
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=2, experiment=experiment.value, shuffle=True, normalize=True, transpose=True, norm_path=f'{output_path}/norm_vals_ANN_F_Exp{experiment.value}_{projection_size}_{gru_hidden_size}_{gru_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv', load_norm=True)

        # Initialize model with proper input dimensions for LSTM
        input_channels = X_train.shape[2]  # Number of features
        # sequence_length = X_train.shape[1]  # Sequence length

        criterion = nn.CrossEntropyLoss()

        model = NeuralNetE(
            input_channels=input_channels,
            projection_size=projection_size,
            output_dim=len(labels),
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            dense1=dense1,
            dense2=dense2,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).squeeze()
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).squeeze()
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        early_stopper = EarlyStopper(patience=es_patience, min_delta=0.001)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        patience_lr = 10     # No improvement for X epochs -> adjust LR
        factor = 0.5              # Factor to multiply/divide LR by
        min_lr = 0.00001
        
        history = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, torch.argmax(y, dim=1))
                total_loss.append(loss.item())
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = np.mean(total_loss)
        
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                # Compute validation loss
                yval_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_loss = criterion(yval_pred, torch.argmax(torch.tensor(y_val, dtype=torch.float32).to(device), dim=1))

                # Compute accuracy
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.argmax(dim=1)).sum().item()
                print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {100 * correct / total}")
        
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_loss': val_loss.item(),
                'accuracy': 100 * correct / total
            })

            # Save best model and adjust learning rate
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f'{output_path}/ANN_F_model_exp{experiment.value}_{projection_size}_{gru_hidden_size}_{gru_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.pth')
                print(f"Model saved at epoch {epoch}")
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
                # Reduce learning rate if no improvement
                if epochs_no_improve >= patience_lr:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = max(old_lr * factor, min_lr)
                        param_group['lr'] = new_lr
                        print(f"Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    
                    epochs_no_improve = 0
                
            # Early stopping check
            if early_stopper.early_stop(val_loss):             
                print(f"Early stopping at epoch {epoch}")
                break
        

        # Save and evaluate model
        model.load_state_dict(torch.load(f'{output_path}/ANN_F_model_exp{experiment.value}_{projection_size}_{gru_hidden_size}_{gru_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.pth'))
        
        df = pd.DataFrame(history)
        
        # Plot training history
        plt.clf()
        plt.plot(df['epoch'], df['loss'], label='loss')
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(f'{output_path}/ANN_F_history_exp{experiment.value}_{projection_size}_{gru_hidden_size}_{gru_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.png')

        # Final predictions
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).squeeze()
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())

        # Save predictions
        y_pred = np.vstack(y_pred_list)
        y_true = np.vstack(y_true_list)
        label_pred = np.argmax(y_pred, axis=1)
        label_true = np.argmax(y_true, axis=1)
        acc = (label_pred == label_true).sum() / len(label_true)
        print(f"Accuracy: {acc}")

        # saving accuracy
        with open(f'{output_path}/ANN_F_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, projection_size {projection_size}, gru_hidden_size {gru_hidden_size}, gru_num_layers {gru_num_layers}, dense1 {dense1}, dense2 {dense2}, dropout {dropout}, lr {lr}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(y_pred, columns=labels)
        df_pred['label_pred'] = label_pred
        df_pred['label_true'] = label_true
        df_pred['id_sample'] = ids

        df_pred.to_csv(f'{output_path}/ANN_F_predictions_exp{experiment.value}_{projection_size}_{gru_hidden_size}_{gru_num_layers}_{dense1}_{dense2}_{dropout}_{lr}_fold{fold}.csv', index=False)

        t2 = time.time()
        print(f'Time: {t2-t1}')


def ann_g_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        d_model=128,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        dropout=0.1,
        epochs: int = 200,
        lr = 0.0001,
        es_patience = 30
        ) -> None:
    """
    Function to run Transformer experiments based on the splits provided.
    """

    set_seed(43)

    device = (
        "cuda:2"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):
        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        # Prepare data for Transformer - note we use dim=2 to get the sequence dimension
        X_train, y_train, X_val, y_val = prepare_data(db_train, label2int, split=0.2, dim=2, experiment=experiment.value, shuffle=True, transpose=True, normalize=True, norm_path=f'{output_path}/norm_vals_ANN_G_Exp{experiment.value}_{d_model}_{num_heads}_{d_ff}_{num_layers}_{dropout}_{lr}_fold{fold}.csv')
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=2, experiment=experiment.value, shuffle=True, normalize=True, transpose=True, norm_path=f'{output_path}/norm_vals_ANN_G_Exp{experiment.value}_{d_model}_{num_heads}_{d_ff}_{num_layers}_{dropout}_{lr}_fold{fold}.csv', load_norm=True)

        # Initialize model with proper input dimensions for LSTM
        input_channels = X_train.shape[2]  # Number of features
        sequence_length = X_train.shape[1]  # Sequence length

        criterion = nn.CrossEntropyLoss()

        model = NeuralNetG(
            input_channels=input_channels,
            seq_len=sequence_length,
            num_classes=len(labels),
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).squeeze()
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).squeeze()
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        early_stopper = EarlyStopper(patience=es_patience, min_delta=0.001)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        patience_lr = 10     # No improvement for X epochs -> adjust LR
        factor = 0.5              # Factor to multiply/divide LR by
        min_lr = 0.00001
        
        history = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, torch.argmax(y, dim=1))
                total_loss.append(loss.item())
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = np.mean(total_loss)
        
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                # Compute validation loss
                yval_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_loss = criterion(yval_pred, torch.argmax(torch.tensor(y_val, dtype=torch.float32).to(device), dim=1))

                # Compute accuracy
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.argmax(dim=1)).sum().item()
                print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {100 * correct / total}")
        
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_loss': val_loss.item(),
                'accuracy': 100 * correct / total
            })

            # Save best model and adjust learning rate
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f'{output_path}/ANN_G_model_exp{experiment.value}_{d_model}_{num_heads}_{d_ff}_{num_layers}_{dropout}_{lr}_fold{fold}.pth')
                print(f"Model saved at epoch {epoch}")
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
                # Reduce learning rate if no improvement
                if epochs_no_improve >= patience_lr:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = max(old_lr * factor, min_lr)
                        param_group['lr'] = new_lr
                        print(f"Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    
                    epochs_no_improve = 0
                
            # Early stopping check
            if early_stopper.early_stop(val_loss):             
                print(f"Early stopping at epoch {epoch}")
                break
        

        # Save and evaluate model
        model.load_state_dict(torch.load(f'{output_path}/ANN_G_model_exp{experiment.value}_{d_model}_{num_heads}_{d_ff}_{num_layers}_{dropout}_{lr}_fold{fold}.pth'))
        
        df = pd.DataFrame(history)
        
        # Plot training history
        plt.clf()
        plt.plot(df['epoch'], df['loss'], label='loss')
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(f'{output_path}/ANN_G_history_exp{experiment.value}_{d_model}_{num_heads}_{d_ff}_{num_layers}_{dropout}_{lr}_fold{fold}.png')

        # Final predictions
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).squeeze()
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())

        # Save predictions
        y_pred = np.vstack(y_pred_list)
        y_true = np.vstack(y_true_list)
        label_pred = np.argmax(y_pred, axis=1)
        label_true = np.argmax(y_true, axis=1)
        acc = (label_pred == label_true).sum() / len(label_true)
        print(f"Accuracy: {acc}")

        # saving accuracy
        with open(f'{output_path}/ANN_G_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, d_model {d_model}, num_heads {num_heads}, d_ff {d_ff}, num_layers {num_layers}, dropout {dropout}, lr {lr}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(y_pred, columns=labels)
        df_pred['label_pred'] = label_pred
        df_pred['label_true'] = label_true
        df_pred['id_sample'] = ids

        df_pred.to_csv(f'{output_path}/ANN_G_predictions_exp{experiment.value}_{d_model}_{num_heads}_{d_ff}_{num_layers}_{dropout}_{lr}_fold{fold}.csv', index=False)

        t2 = time.time()
        print(f'Time: {t2-t1}')


def rf_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1,
        max_samples=None,
        criterion='gini',
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        random_state=42
        ) -> None:
    """
    Function to run ANN experiments based on the splits provided.

    Parameters
    ----------
    db_path : str
        Path to the database.
    splits_path : str
        Path to the splits.
    output_path : str
        Path to save the results.
    experiment : int
        Kin of experiment to run (1, 2, 3):
            1: Bands
            2: Spectral indices
            3: Bands + Spectral indices
    

    Returns
    -------
    None
    """

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):

        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        X_train, y_train, _, _ = prepare_data(db_train, label2int, dim=1, experiment=experiment.value, shuffle=True)
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=1, experiment=experiment.value, shuffle=True)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=True,
            max_samples=max_samples,
            criterion=criterion,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, np.argmax(y_train, axis = 1))

        # Evaluate the model
        y_pred = model.predict_proba(X_test)
        # get the index of the max probability
        y_pred2 = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)


        acc = (y_pred2 == y_test).sum() / len(y_test)
        print(f"Accuracy: {acc}")

        # saving accuracy
        with open(f'{output_path}/rf_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, n_estimators {n_estimators}, max_depth {max_depth}, min_samples_split {min_samples_split}, min_samples_leaf {min_samples_leaf}, max_features {max_features}, max_samples {max_samples}, criterion {criterion}, min_impurity_decrease {min_impurity_decrease}, ccp_alpha {ccp_alpha}, random_state {random_state}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(
            {
                'id_sample':ids,
                'label_pred': y_pred2,
                'label_true': y_test
            }
        )

        df_pred = pd.concat([df_pred,pd.DataFrame(y_pred)], axis=1)

        # Save the model and results
        joblib.dump(model, f'{output_path}/rf_model_{experiment.value}_{n_estimators}_{max_depth}_{min_samples_split}_{min_samples_leaf}_{max_features}_{max_samples}_{criterion}_{min_impurity_decrease}_{ccp_alpha}_{random_state}_fold{fold}.pkl')
        df_pred.to_csv(f'{output_path}/rf_predictions_{experiment.value}_{n_estimators}_{max_depth}_{min_samples_split}_{min_samples_leaf}_{max_features}_{max_samples}_{criterion}_{min_impurity_decrease}_{ccp_alpha}_{random_state}_fold{fold}.csv', index=False)
        
        t2 = time.time()
        print(f'Time: {t2-t1}')


def xgb_experiment(
        db_path: str,
        splits_path: str,
        experiment: int,
        output_path: str,
        n_estimators=5000,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        learning_rate=0.03,
        random_state=42
        ) -> None:
    """
    Function to run ANN experiments based on the splits provided.

    Parameters
    ----------
    db_path : str
        Path to the database.
    splits_path : str
        Path to the splits.
    output_path : str
        Path to save the results.
    experiment : int
        Kin of experiment to run (1, 2, 3):
            1: Bands
            2: Spectral indices
            3: Bands + Spectral indices
    

    Returns
    -------
    None
    """

    experiment = Experiment(experiment)

    db = pd.read_csv(db_path)
    dfsplitted = pd.read_csv(splits_path)

    print(f'Starting experiment {experiment.name}...')

    for fold in range(3):

        t1 = time.time()

        dffold = dfsplitted[dfsplitted['fold'] == fold]

        train = dffold[dffold['split'] == 'train']['id']
        test = dffold[dffold['split'] == 'test']['id']

        db_train = db[db['id'].isin(train)].drop(columns=['date'])
        db_test = db[db['id'].isin(test)].drop(columns=['date'])

        labels = sorted(db['class'].unique())
        label2int = {label: i for i, label in enumerate(labels)}

        X_train, y_train, X_val, y_val = prepare_data(db_train, label2int, split=0.2, dim=1, experiment=experiment.value, shuffle=True)
        X_test, y_test, ids, _ = prepare_data(db_test, label2int, dim=1, experiment=experiment.value, shuffle=True)

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(labels),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            tree_method="auto",
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
            device='cuda:4',
            early_stopping_rounds=50,
        )

        model.fit(X_train, np.argmax(y_train, axis = 1),
                  eval_set=[(X_val, np.argmax(y_val, axis=1))],
                  verbose=100
        )

        # Evaluate the model
        y_pred = model.predict_proba(X_test)
        # get the index of the max probability
        y_pred2 = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)


        acc = (y_pred2 == y_test).sum() / len(y_test)
        print(f"Accuracy: {acc}")

        # saving accuracy
        with open(f'{output_path}/xgb_acc.txt', 'a') as f:
            f.write(f'Experiment {experiment.value}, fold {fold}, n_estimators {n_estimators}, max_depth {max_depth}, min_child_weight {min_child_weight}, subsample {subsample}, colsample_bytree {colsample_bytree}, reg_lambda {reg_lambda}, reg_alpha {reg_alpha}, learning_rate {learning_rate}, random_state {random_state}, Accuracy {acc}\n')

        df_pred = pd.DataFrame(
            {
                'id_sample':ids,
                'label_pred': y_pred2,
                'label_true': y_test
            }
        )

        df_pred = pd.concat([df_pred,pd.DataFrame(y_pred)], axis=1)

        # Save the model and results
        joblib.dump(model, f'{output_path}/xgb_model_{experiment.value}_{n_estimators}_{max_depth}_{min_child_weight}_{subsample}_{colsample_bytree}_{reg_lambda}_{reg_alpha}_{learning_rate}_{random_state}_fold{fold}.pkl')
        df_pred.to_csv(f'{output_path}/xgb_predictions_{experiment.value}_{n_estimators}_{max_depth}_{min_child_weight}_{subsample}_{colsample_bytree}_{reg_lambda}_{reg_alpha}_{learning_rate}_{random_state}_fold{fold}.csv', index=False)
        
        t2 = time.time()
        print(f'Time: {t2-t1}')
