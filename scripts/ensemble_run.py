import rioxarray as rxr
import xarray as xr
import os

path = 'scenes_final'
files = os.listdir(path)
files = [f for f in files if (f.endswith('.tif')) and ('predictions' in f)]
files = sorted(files)
cells = [v.split('_')[0] for v in files]
folds = [int(v.split('_')[1]) for v in files]

### approach: product of experts
l_class = []
l_prob = []

for cell in sorted(set(cells)):
    files_selected = [f for f, c in zip(files, cells) if c == cell]
    
    l = [rxr.open_rasterio(os.path.join(path, f)) for f in files_selected]
    
    for i in range(3):
        l[i] = l[i].clip(1e-7, 1)
        
    ds = l[0] * l[1] * l[2] 
    
    ds = ds/ds.sum(dim='band')
    
    ds_class = ds.argmax(dim='band')
    ds_maxprob = ds.max(dim='band')

    ds_class.rio.to_raster(os.path.join(path, f'{cell}_mosaic_classes.tif'))
    ds_maxprob.rio.to_raster(os.path.join(path, f'{cell}_mosaic_maxprob.tif'))

### approach: mean prob
l_class = []
l_prob = []

for cell in sorted(set(cells)):
    files_selected = [f for f, c in zip(files, cells) if c == cell]
    
    l = [rxr.open_rasterio(os.path.join(path, f)) for f in files_selected]
        
    ds = (l[0] + l[1] + l[2])/3
    
    ds = ds/ds.sum(dim='band')
    
    ds_class = ds.argmax(dim='band')
    ds_maxprob = ds.max(dim='band')

    ds_class.rio.to_raster(os.path.join(path, f'{cell}_mosaic_classesOpt2.tif'))
    ds_maxprob.rio.to_raster(os.path.join(path, f'{cell}_mosaic_maxprobOpt2.tif'))

### approach: max prob
l_class = []
l_prob = []

for cell in sorted(set(cells)):
    files_selected = [f for f, c in zip(files, cells) if c == cell]
    
    l = [rxr.open_rasterio(os.path.join(path, f)) for f in files_selected]
        
    stacked = xr.concat([l[0], l[1], l[2]], dim='fold')
    
    ds = stacked.max(dim='fold')
    ds = ds/ds.sum(dim='band')
    
    ds_class = ds.argmax(dim='band')
    ds_maxprob = ds.max(dim='band')

    ds_class.rio.to_raster(os.path.join(path, f'{cell}_mosaic_classesOpt3.tif'))
    ds_maxprob.rio.to_raster(os.path.join(path, f'{cell}_mosaic_maxprobOpt3.tif'))