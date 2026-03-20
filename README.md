# S2-ClassifierComparison

An exhaustive comparison between DTW, neural networks, and decision tree ensembles for crop classification using Sentinel-2 time series data.

## Overview

This repository implements and compares multiple classification approaches for crop type mapping from Sentinel-2 satellite imagery. Each method is evaluated on three feature variants (raw spectral bands, spectral indices, and combined features) using 3-fold cross-validation. Functional ANOVA (fANOVA) is used to analyze hyperparameter importance for each model. Trained models can be applied to full Sentinel-2 scenes to produce classification and probability rasters.

## Methods

### Distance-Based
| Method | Script | Description |
|--------|--------|-------------|
| DTW | `scripts/script_dtw.py` | Dynamic Time Warping and Time-Weighted DTW with logistic temporal cost|

### Neural Networks
| Method | Script | Architecture |
|--------|--------|-------------|
| ANN-A | `scripts/script_ann_exp_a.py` | 3-layer fully connected network |
| ANN-B | `scripts/script_ann_exp_b.py` | 1D CNN (2 conv layers) + 3 dense layers |
| ANN-C | `scripts/script_ann_exp_c.py` | Fully connected with PCA preprocessing (15 components) |
| ANN-D | `scripts/script_ann_exp_d.py` | 1D CNN (3 conv layers) + 2 dense layers |
| ANN-E | `scripts/script_ann_exp_e.py` | Bidirectional LSTM |
| ANN-F | `scripts/script_ann_exp_f.py` | Bidirectional GRU |
| ANN-G | `scripts/script_ann_exp_g.py` | Transformer encoder |

### Tree-Based Ensembles
| Method | Script | Description |
|--------|--------|-------------|
| Random Forest | `scripts/script_rf.py` | sklearn RandomForestClassifier |
| XGBoost | `scripts/script_xgb.py` | Gradient boosting (GPU-accelerated) |

## Data

The dataset is Sentinel-2 time series with **69 time steps** covering **16 crop classes**.

### Input Files
- `./data/data_points_filled.csv` — Time series with columns: `date`, `id`, `class`, `B02`, `B03`, `B04`, `B08`
- `./data/splitted_data.csv` — Pre-defined train/test splits with columns: `id`, `split` (`train`/`test`), `fold` (0, 1, 2)

### Feature Variants (Experiments)
Each classifier is run under three feature configurations:

| Experiment | Features |
|------------|----------|
| Exp 1 | Raw Sentinel-2 bands: B02, B03, B04, B08 |
| Exp 2 | Spectral indices: NDVI, GRVI, GNDVI, EVI |
| Exp 3 | Combined: all bands + all indices (8 features) |

### Example Scene Data
The `scenes_example/` directory contains example Sentinel-2 GeoTIFF files for testing the inference pipeline:   
- `B02.tif`, `B03.tif`, `B04.tif`, `B08.tif` — Individual band rasters for a spatial subset

### Output Files
- `./data/final_classification_map.tif` — Final crop classification map produced by the inference pipeline


## Usage

### 1. Training Classifiers

All training scripts live under `scripts/` and are self-contained. Each handles the three feature experiments automatically and skips previously completed runs:

```bash
cd scripts

python script_rf.py
python script_xgb.py
python script_dtw.py
python script_ann_exp_a.py
python script_ann_exp_b.py
python script_ann_exp_c.py
python script_ann_exp_d.py
python script_ann_exp_e.py
python script_ann_exp_f.py
python script_ann_exp_g.py
```

> **Note:** XGBoost uses `device='cuda:4'` by default. Adjust this if your GPU setup differs.

### 2. Scene Inference

Apply trained models to Sentinel-2 scenes to produce per-pixel classification and probability rasters.

#### Per-fold inference

`scripts/inference_ann.py` runs a single trained model over all scene cells in an input directory and outputs a probability raster (16 bands, one per crop class) for each cell:

```bash
python scripts/inference_ann.py \
  --model_path final_models/ANN_G_model_exp1_256_8_128_1_0.3_0.001_fold0.pth \
  --norm_path final_models/norm_vals_ANN_G_Exp1_256_8_128_1_0.3_0.001_fold0.csv \
  --seq_len 69 \
  --classes 16 \
  --input_dir scenes_filled \
  --output_dir scenes_final
```

Output files: `{cell}_{fold}_predictions.tif` — multi-band probability rasters

#### Multi-fold ensemble

`scripts/ensemble_run.py` combines the per-fold probability rasters from `scenes_final/` (not included due to raster size) using three ensemble strategies and produces final classification and confidence maps:

```bash
python scripts/ensemble_run.py
```

| Strategy | Class map | Confidence map |
|----------|-----------|----------------|
| Product-of-Experts | `{cell}_mosaic_classesOpt1.tif` | `{cell}_mosaic_maxprobOpt1.tif` |
| Mean probability | `{cell}_mosaic_classesOpt2.tif` | `{cell}_mosaic_maxprobOpt2.tif` |
| Max probability | `{cell}_mosaic_classesOpt3.tif` | `{cell}_mosaic_maxprobOpt3.tif` |

#### Quick example

`classification_example.py` provides a self-contained end-to-end example using the three ANN-G (Transformer) fold models and the `scenes_example/` data:

```bash
python classification_example.py
```

Outputs: `mosaic_classes.tif` (class predictions) and `mosaic_maxprob.tif` (confidence scores).

### 3. Running fANOVA

After training, analyze which hyperparameters most influence accuracy:

```bash
cd fanova

# Transformer
python run.py -p ../data/ann_g/ANN_G_acc.txt -n transformer \
  --hp d_model num_heads d_ff num_layers dropout lr -o ./outputs

# Bidirectional LSTM
python run.py -p ../data/ann_e/ANN_E_acc.txt -n lstm \
  --hp projection_size dense1 dense2 lstm_hidden_size lstm_num_layers lr dropout -o ./outputs

# Bidirectional GRU
python run.py -p ../data/ann_f/ANN_F_acc.txt -n gru \
  --hp projection_size dense1 dense2 gru_hidden_size gru_num_layers lr dropout -o ./outputs

# 1D CNN (ANN-B)
python run.py -p ../data/ann_b/ANN_B_acc.txt -n cnn_b \
  --hp conv1_channels conv2_channels dense1 dense2 dense3 lr dropout -o ./outputs

# 1D CNN (ANN-D)
python run.py -p ../data/ann_d/ANN_D_acc.txt -n cnn_d \
  --hp conv1_channels conv2_channels conv3_channels dense1 dense2 lr dropout -o ./outputs

# Random Forest
python run.py -p ../data/rf/rf_acc.txt -n rf \
  --hp n_estimators max_features criterion min_impurity_decrease ccp_alpha -o ./outputs

# XGBoost
python run.py -p ../data/xgb/xgb_acc.txt -n xgb \
  --hp n_estimators max_depth min_child_weight learning_rate subsample colsample_bytree reg_lambda reg_alpha -o ./outputs
```

#### fANOVA Output Files

| File | Description |
|------|-------------|
| `{name}_fanova_main_effects.csv` | Individual hyperparameter importance scores |
| `{name}_fanova_pairwise_interactions.csv` | Pairwise interaction importance matrix |
| `{name}_fanova_main_effects.pdf` | Bar chart of top hyperparameter importances |
| `{name}_fanova_pairwise_interactions.pdf` | Heatmap of pairwise interactions |
| `{name}_fanova_total_variance.txt` | Summary: RMSE, R², mean accuracy, quantiles |

## Hyperparameters

### Neural Networks

| Model | Hyperparameter | Range / Options |
|-------|---------------|-----------------|
| ANN-A, ANN-C | `l1`, `l2`, `l3` (layer sizes) | 16–512 |
| ANN-B | `conv1_channels`, `conv2_channels` | 16–128 |
| ANN-D | `conv1_channels`, `conv2_channels`, `conv3_channels` | 16–128 |
| ANN-E | `lstm_hidden_size`, `lstm_num_layers` | 32–256, 1–3 |
| ANN-F | `gru_hidden_size`, `gru_num_layers` | 32–256, 1–3 |
| ANN-G | `d_model`, `num_heads`, `d_ff`, `num_layers` | 128/256, 2/4/8/16, 128–512, 1/2/4/8 |
| All ANNs | `dropout` | 0.0–0.3 |
| All ANNs | `lr` (learning rate) | 0.001, 0.0001 |
| ANN-E, ANN-F | `projection_size` | 0, 64, 128 |

### Tree-Based

| Model | Hyperparameter | Range / Options |
|-------|---------------|-----------------|
| Random Forest | `n_estimators` | 50–400 |
| Random Forest | `max_features` | `sqrt`, `log2`, `None` |
| Random Forest | `criterion` | `gini`, `entropy` |
| Random Forest | `min_impurity_decrease` | 0.0–0.05 |
| Random Forest | `ccp_alpha` | 0.0–0.05 |
| XGBoost | `n_estimators` | 100–2000 |
| XGBoost | `max_depth` | 3, 5, 9, 12 |
| XGBoost | `min_child_weight` | 1, 5, 10 |
| XGBoost | `learning_rate` | 0.01–0.2 |
| XGBoost | `subsample`, `colsample_bytree` | 0.5–1.0 |
| XGBoost | `reg_lambda`, `reg_alpha` | 0.0–1.0 |

### TWDTW

| Configuration | `alpha` | `beta` |
|--------------|---------|--------|
| Config 1 | 0.1 | 100 |
| Config 2 | 0.1 | 50 |
| Config 3 | 0.05 | 35 |

## Training Details

- **Cross-validation:** 3-fold (folds defined in `splitted_data.csv`)
- **Optimization:** Adam optimizer with dynamic learning rate reduction
- **Early stopping:** Patience of 30 epochs (neural networks); 50 rounds (XGBoost)
- **Epochs:** Up to 200 (neural networks)
- **Train/validation split:** 80/20 *within training fold* (neural networks and XGBoost)
- **Reproducibility:** Fixed random seeds via `set_seed()`
- **DTW parallelism:** Dask-based parallel distance matrix computation

## Dependencies

Core dependencies:

```
torch
scikit-learn
xgboost
dask
rasterio
pyrfr
pandas
numpy
matplotlib
```
