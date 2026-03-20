import os
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import scipy.special as sp
import xarray as xr
import rioxarray as rio
from src.s2c.ann import NeuralNetG

def s(x):
    """
    Sigmoid activation function
    0 <= s(x) <= 1
    """
    return sp.expit(x)

model_paths = ["final_models/ANN_G_model_exp1_256_8_128_1_0.3_0.001_fold0.pth",
               "final_models/ANN_G_model_exp1_256_8_128_1_0.3_0.001_fold1.pth",
               "final_models/ANN_G_model_exp1_256_8_128_1_0.3_0.001_fold2.pth"]

norm_paths = ["final_models/norm_vals_ANN_G_Exp1_256_8_128_1_0.3_0.001_fold0.csv",
              "final_models/norm_vals_ANN_G_Exp1_256_8_128_1_0.3_0.001_fold1.csv",
              "final_models/norm_vals_ANN_G_Exp1_256_8_128_1_0.3_0.001_fold2.csv"]

files = ["scenes_example/B02.tif",
         "scenes_example/B03.tif",
         "scenes_example/B04.tif",
         "scenes_example/B08.tif"]

num_classes = 16
seq_len = 69
opath = '.'
channels = 4

processed_folds = []

for i in range(len(model_paths)):
    model_path = model_paths[i]
    norm_path = norm_paths[i]
    name_list = model_path.split('/')[-1].split('_')
    norm = pd.read_csv(norm_path)
    d_model = int(name_list[4])
    num_heads = int(name_list[5])
    dim_feedforward = int(name_list[6])
    num_layers = int(name_list[7])
    dropout = float(name_list[8])
    learning_rate = float(name_list[9])
    fold = int(name_list[10].split('.')[0][-1])
    print(f'Processing fold {fold} with model {model_path}')

    l = [xr.open_dataarray(val) for val in files]
    row = l[1].shape[1]
    col = l[2].shape[2]
    
    print(f'Normalizing data...')
    l = [(l[i]/10000 - norm.loc[i, 'mean']) / norm.loc[i, 'std'] for i in range(len(l))]
    
    print(f'Concatenating data...')
    ds = xr.concat(l, dim='time')
    
    print(f'Getting numpy array...')
    dl = ds.to_numpy()
    
    print(f'Reshaping data...')
    dl = dl.reshape(channels, seq_len, row*col) ###
    
    print(f'Moving axis...')
    dl = np.moveaxis(dl, 2, 0)
    dl = np.moveaxis(dl, 2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype=torch.float32
    
    model = NeuralNetG(
        input_channels=channels,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=dim_feedforward,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    print(f'Loading model weights...')
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
    model = model.to(device, dtype=dtype)
    
    print(f'Creating DataLoader...')
    dl2 = DataLoader(dl, batch_size=4096, shuffle=False)
    
    print(f'Running inference...')
    y_pred = np.empty((col*row, num_classes), dtype=np.float32)
    
    for i, batch in enumerate(dl2):
        x = batch.to(device, dtype=dtype)
        model.eval()
        with torch.no_grad():
            yhat = model(x)
        yhat = yhat.cpu().numpy()
        yhat = s(yhat) # sigmoid activation
        for j in range(num_classes):
            temp = yhat[:, j]/np.sum(yhat, axis=1) # softmax normalization
            y_pred[i*4096:(i+1)*4096, j] = temp
        print(f'Batch {i}')
    
    print(f'Processing predictions...')
    r = l[0].isel(band=0).copy()
    ypred2 = y_pred.reshape(row, col, num_classes)
    l2 = []
    for j in range(num_classes):
        r2 = r.copy()
        r2.values = ypred2[:, :, j]
        r2.band.values = j
        l2.append(r2)
    r_final = xr.concat(l2, dim='band')
    if "long_name" in r_final.attrs:
        del r_final.attrs["long_name"]
    
    processed_folds.append(r_final)

# Ensamble by product of experts approach
l_class = []
l_prob = []

for i in range(3):
    processed_folds[i] = processed_folds[i].clip(1e-7, 1) # avoid zero probabilities, adding a small constant
    
ds = processed_folds[0] * processed_folds[1] * processed_folds[2] 

ds = ds/ds.sum(dim='band')

ds_class = ds.argmax(dim='band')
ds_maxprob = ds.max(dim='band')

ds_class.rio.to_raster(os.path.join(opath, f'mosaic_classes.tif'))
ds_maxprob.rio.to_raster(os.path.join(opath, f'mosaic_maxprob.tif'))