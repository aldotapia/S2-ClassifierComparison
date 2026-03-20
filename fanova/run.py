# Examples run
# Usage:
# python run.py -p ~/S2-ClassifierComparision/data/ann_g/ANN_G_acc.txt -n transformer --hp d_model num_heads d_ff num_layers dropout lr -o ./outputs
# python run.py -p ~/S2-ClassifierComparision/data/rf/rf_acc.txt -n rf --hp n_estimators max_features criterion min_impurity_decrease ccp_alpha -o ./outputs

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from pyrfr import regression as reg

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)

from fanova import fANOVA

from utils import *

import argparse

import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()

argparser.add_argument('-p', '--path', type=str, required=True, help='Path to results file')
argparser.add_argument('-n', '--name', type=str, required=True, help='Name for the output files')
argparser.add_argument('--hp', nargs='+', type=str, required=True, help='Hyperparameter names')
argparser.add_argument('-o', '--out', type=str, required=True, help='Output directory')
args = argparser.parse_args()

df = read_results(args.path)
    
df = df.groupby(args.hp).mean().reset_index()
df = df.drop(columns=['Experiment','fold'])

target_col = "Accuracy"
param_cols = [c for c in df.columns if c != target_col]

X_df = df[param_cols].copy()
y = df[target_col].to_numpy(dtype=float)

X_enc, meta = encode_params_df(X_df)

Xtr, Xte, ytr, yte = train_test_split(X_enc, y, test_size=0.2, random_state=0)

data_tr = build_pyrfr_container(Xtr, ytr, meta)
data_all = build_pyrfr_container(X_enc, y, meta)

rng = make_rng(42)
n_train = Xtr.shape[0]

candidates = []
for num_trees in [32, 64, 128]:
    for max_depth in [16, 32, 64]:
        for min_split in [2, 3, 5]:
            for min_leaf in [1, 3, 5]:
                    candidates.append(
                        dict(
                            num_trees=num_trees,
                            max_depth=max_depth,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            do_bootstrapping=True,
                            epsilon_purity=1e-8,
                        )
                    )

def predict_forest(forest, X: np.ndarray):
    return np.array([forest.predict(x.tolist()) for x in X], dtype=float)

best = None
best_rmse = np.inf

for params in candidates:
    forest = reg.fanova_forest()
    forest = configure_fanova_forest(forest, params)
    forest = set_points_per_tree(forest, n_train)

    forest.fit(data_tr, rng)

    yhat = predict_forest(forest, Xte)
    rmse = mean_squared_error(yte, yhat, squared=False)

    if rmse < best_rmse:
        best_rmse = rmse
        best = dict(params=params, forest=forest, rmse=rmse, r2=r2_score(yte, yhat))

print("RMSE:", best["rmse"])
print("R2:", best["r2"])


p = best["params"]

# redoing this part to prepare for fANOVA
hp_names = [c for c in df.columns if c != target_col]

hp_names = sorted(hp_names) 

X_df = df[hp_names].copy()
y = df[target_col].to_numpy(dtype=float)


cs = ConfigurationSpace(seed=42)

cat_maps = {}
for col in hp_names:
    s = X_df[col].dropna()

    if pd.api.types.is_object_dtype(s) or str(s.dtype) == "category" or pd.api.types.is_bool_dtype(s):
        cat = s.astype("category")
        levels = list(cat.cat.categories)
        cs.add(CategoricalHyperparameter(col, levels))
        cat_maps[col] = {v: i for i, v in enumerate(levels)}
        continue

    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.isna().any():
        bad = s[s_num.isna()].unique()[:5]
        raise ValueError(f"Column '{col}' has non-numeric values like {bad}.")

    is_int_like = (s_num % 1 == 0).all()
    lo, hi = float(s_num.min()), float(s_num.max())

    if np.isclose(lo, hi):
        cs.add(CategoricalHyperparameter(col, [int(round(lo))] if is_int_like else [float(lo)]))
        continue

    if is_int_like:
        cs.add(UniformIntegerHyperparameter(col, int(np.floor(lo)), int(np.ceil(hi))))
    else:
        is_log = (col.lower() in ["lr", "learning_rate", "weight_decay"]) and lo > 0 and hi > 0
        cs.add(UniformFloatHyperparameter(col, float(lo), float(hi), log=is_log))

cs_order = [hp.name for hp in list(cs.values())]

X_fanova = np.zeros((len(X_df), len(cs_order)), dtype=float)

for j, name in enumerate(cs_order):
    if name in cat_maps:
        # map category value -> code
        mapper = cat_maps[name]
        X_fanova[:, j] = X_df[name].map(mapper).to_numpy(dtype=float)
    else:
        X_fanova[:, j] = pd.to_numeric(X_df[name], errors="raise").to_numpy(dtype=float)


for j, hp in enumerate(list(cs.values())):
    xj = X_fanova[:, j]
    if isinstance(hp, (UniformIntegerHyperparameter, UniformFloatHyperparameter)):
        if xj.min() < hp.lower - 1e-12 or xj.max() > hp.upper + 1e-12:
            raise RuntimeError(f"Out of bounds for {hp.name}: {xj.min()}..{xj.max()} vs {hp.lower}..{hp.upper}")


f = fANOVA(
    X_fanova, y, cs,
    n_trees=int(p.get("num_trees")),
    seed=42,
    bootstrapping=bool(p.get("do_bootstrapping")),
    points_per_tree=int(X_fanova.shape[0]),
    min_samples_split=int(p.get("min_samples_split")),
    min_samples_leaf=int(p.get("min_samples_leaf")),
    max_depth=int(p.get("max_depth")),
)


main = {}
for i, name in enumerate(cs_order):
    d = f.quantify_importance((i,))
    main[name] = d[(i,)]['individual importance']

main = dict(sorted(main.items(), key=lambda kv: kv[1], reverse=True))
print("Main effects:")
for k, v in main.items():
    print(f"{k:12s} {v:.4f}")

pd.Series(main).to_csv(f'{args.out}/{args.name}_fanova_main_effects.csv')

pairwise_mar = f.get_most_important_pairwise_marginals()

l = []

for item, value in pairwise_mar.items():
    l.append({'item1': item[0], 'item2': item[1], 'importance': value})

pd.DataFrame(l).to_csv(f'{args.out}/{args.name}_fanova_pairwise_interactions.csv', index=False)

 uncom,ment later
main_s = pd.Series(main).sort_values(ascending=False)

top_k = min(15, len(main_s))
main_top = main_s.head(top_k)[::-1]  # highest interaction first

plt.figure()
plt.barh(main_top.index, main_top.values)
plt.xlabel("fANOVA main effect importance")
plt.title(f"Most sensitive hyperparameters")
plt.tight_layout()
plt.savefig(f'{args.out}/{args.name}_fanova_main_effects.pdf', bbox_inches='tight')


top_k = min(10, len(main_s))
top_params = list(main_s.head(top_k).index)
top_idx = [cs_order.index(p) for p in top_params]

M = np.zeros((top_k, top_k), dtype=float)


for a, i in enumerate(top_idx):
    for b, j in enumerate(top_idx):
        if a == b:
            M[a, b] = 0.0
        elif a < b:
            d = f.quantify_importance((i, j))
            M[a, b] = d[(i, j)]['individual importance']
            M[b, a] = M[a, b]

plt.figure()
plt.imshow(M, aspect="auto")
plt.xticks(range(top_k), top_params, rotation=45, ha="right")
plt.yticks(range(top_k), top_params)
plt.colorbar(label="pairwise interaction importance")
plt.title("Pairwise interactions (fANOVA)")
plt.tight_layout()
plt.savefig(f'{args.out}/{args.name}_fanova_pairwise_interactions.pdf', bbox_inches='tight')

# mean total variance
mean_var = np.mean(np.asarray(f.trees_total_variance))
# std total variance
std_var = np.std(np.asarray(f.trees_total_variance))

# mean accuracy from original data
mean_acc = df['Accuracy'].mean()
std_acc = df['Accuracy'].std()

# min and max
min_acc = df['Accuracy'].min()
max_acc = df['Accuracy'].max()

# 10 and 90 quantiles
q10 = df['Accuracy'].quantile(0.1)
q90 = df['Accuracy'].quantile(0.9)

q05 = df['Accuracy'].quantile(0.05)
q95 = df['Accuracy'].quantile(0.95)


with open(f'{args.out}/{args.name}_fanova_total_variance.txt', 'w') as f_out:
    f_out.write(f"{mean_var}\n")
    f_out.write(f"{std_var}\n")
    f_out.write(f"{best['rmse']}\n")
    f_out.write(f"{best['r2']}\n")
    f_out.write(f"{mean_acc}\n")
    f_out.write(f"{std_acc}\n")
    f_out.write(f"{min_acc}\n")
    f_out.write(f"{max_acc}\n")
    f_out.write(f"{q05}\n")
    f_out.write(f"{q95}\n")
    f_out.write(f"{q10}\n")
    f_out.write(f"{q90}\n")