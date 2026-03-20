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

# example
# python inference_ann2.py --model_path final_models/ANN_G_model_exp1_256_8_128_1_0.3_0.001_fold0.pth --norm_path final_models/norm_vals_ANN_G_Exp1_256_8_128_1_0.3_0.001_fold0.csv --seq_len 69 --classes 16 --input_dir scenes_filled --output_dir scenes_final
# python inference_ann2.py --model_path final_models/ANN_G_model_exp1_256_8_128_1_0.3_0.001_fold1.pth --norm_path final_models/norm_vals_ANN_G_Exp1_256_8_128_1_0.3_0.001_fold1.csv --seq_len 69 --classes 16 --input_dir scenes_filled --output_dir scenes_final
# python inference_ann2.py --model_path final_models/ANN_G_model_exp1_256_8_128_1_0.3_0.001_fold2.pth --norm_path final_models/norm_vals_ANN_G_Exp1_256_8_128_1_0.3_0.001_fold2.csv --seq_len 69 --classes 16 --input_dir scenes_filled --output_dir scenes_final

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', type=str, default='', help='Path to the model file')
argparser.add_argument('--norm_path', type=str, default='', help='Path to normalization values CSV file')
argparser.add_argument('--seq_len', type=int, default=69, help='Sequence length')
argparser.add_argument('--classes', type=int, default=16, help='Number of input channels')
argparser.add_argument('--input_dir', type=str, default='/path/to/input', help='Directory with input data')
argparser.add_argument('--output_dir', type=str, default='/path/to/output', help='Directory to save output predictions')
args = argparser.parse_args()

model_path = args.model_path
norm_path = args.norm_path
fpath = args.input_dir
opath = args.output_dir
files = os.listdir(fpath)
files = sorted(files)

seq_len = args.seq_len
num_classes = args.classes

name_list = model_path.split('/')[-1].split('_')
norm = pd.read_csv(norm_path)

channels = 4 if name_list[3] != 'exp3' else 8
d_model = int(name_list[4])
num_heads = int(name_list[5])
dim_feedforward = int(name_list[6])
num_layers = int(name_list[7])
dropout = float(name_list[8])
learning_rate = float(name_list[9])
fold = int(name_list[10].split('.')[0][-1])

cells = [val.split('_')[0] for val in files if val.endswith('.tif')]
cells = sorted(list(set(cells)))

def s(x):
    """
    Sigmoid activation function
    0 <= s(x) <= 1
    """
    return sp.expit(x)

def process_cell(cell):
    """Process a single cell with Dask delayed decorator"""
    output_file = f'{opath}/{cell}_{fold}_predictions.tif'
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f'Cell {cell}: Output file already exists. Skipping...')
        return f'Skipped {cell}'
    
    print(f'Cell {cell}: Processing cell')
    
    files2 = [val for val in files if val.startswith(f'{cell}_') and val.endswith('.tif')]
    
    if not files2:
        print(f'Cell {cell}: No files found. Skipping...')
        return f'No files for {cell}'

    print(f'Cell {cell}: Loading data...')
    l = [xr.open_dataarray(fpath + '/' + val) for val in files2]

    row = l[1].shape[1]
    col = l[2].shape[2]

    print(f'Cell {cell}: Normalizing data...')
    l = [(l[i]/10000 - norm.loc[i, 'mean']) / norm.loc[i, 'std'] for i in range(len(l))]

    print(f'Cell {cell}: Concatenating data...')
    ds = xr.concat(l, dim='time')

    print(f'Cell {cell}: Getting numpy array...')
    dl = ds.to_numpy()

    print(f'Cell {cell}: Reshaping data...')
    dl = dl.reshape(channels, seq_len, row*col) ###

    print(f'Cell {cell}: Moving axis...')
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
    
    print(f'Cell {cell}: Loading model weights...')
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
    model = model.to(device, dtype=dtype)

    print(f'Cell {cell}: Creating DataLoader...')
    dl2 = DataLoader(dl, batch_size=4096, shuffle=False)

    print(f'Cell {cell}: Running inference...')
    y_pred = np.empty((col*row, num_classes), dtype=np.float32) ####
    
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
        print(f'Cell {cell}: Batch {i}')
    
    print(f'Cell {cell}: Saving predictions...')
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
    
    r_final.rio.to_raster(output_file, driver='GTiff')
    
    return f'Completed {cell}'

def main():
    # Create output directory if it doesn't exist
    os.makedirs(opath, exist_ok=True)
    
    print(f'Processing {len(cells)} cells with Dask parallelization...')

    for cell in cells:
        print(f'Scheduling processing for cell: {cell}')
        process_cell(cell)

if __name__ == "__main__":
    main()
    print('Inference completed successfully.')
    print('All predictions saved to:', opath)