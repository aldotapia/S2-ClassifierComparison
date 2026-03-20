import numpy as np
import pandas as pd
from pyrfr import regression as reg

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

def encode_params_df(X_df: pd.DataFrame):
    """
    Returns:
      X_enc: float64 ndarray (n, d), with categorical columns encoded as int codes
      meta:  list of dicts, one per feature:
             {"name": str, "kind": "num"|"cat", "bounds": (min,max) or None, "cats": [levels] or None}
    """
    X_work = X_df.copy()
    meta = []

    for col in X_work.columns:
        if (X_work[col].dtype == "object") or (str(X_work[col].dtype) == "category"):
            # categorical -> integer codes 0..K-1
            cat = X_work[col].astype("category")
            levels = list(cat.cat.categories)
            codes = cat.cat.codes.astype(np.int64)

            # If there are missing values, cat.cat.codes gives -1 for NaN.
            # pyrfr categorical must be in 0..K-1, so either drop NaNs or impute.
            if (codes < 0).any():
                # simplest: impute missing with most frequent category
                mode = cat.mode(dropna=True).iloc[0]
                cat2 = X_work[col].fillna(mode).astype("category")
                # keep same levels if possible
                cat2 = cat2.cat.set_categories(levels)
                codes = cat2.cat.codes.astype(np.int64)

            X_work[col] = codes
            meta.append({"name": col, "kind": "cat", "cats": levels, "bounds": None})
        else:
            # numeric
            vals = pd.to_numeric(X_work[col], errors="coerce")
            # impute numeric missing with median
            med = float(np.nanmedian(vals.to_numpy()))
            vals = vals.fillna(med).astype(float)
            X_work[col] = vals

            mn = float(vals.min())
            mx = float(vals.max())
            if mn == mx:
                # constant feature; fANOVA can return NaNs or 0 importance for constants.
                # keep a tiny range so bounds are valid
                mx = mn + 1e-12

            meta.append({"name": col, "kind": "num", "cats": None, "bounds": (mn, mx)})

    X_enc = X_work.to_numpy(dtype=np.float64)
    return X_enc, meta


def build_pyrfr_container(X: np.ndarray, y: np.ndarray, meta):
    """
    Build pyrfr.regression.default_data_container and set:
      - type of each feature (0 numeric, K categorical)
      - bounds for numeric features (needed for marginalization)
      - add_data_point(features, response)
    """
    n, d = X.shape
    data = reg.default_data_container(d)

    # specify feature types + bounds
    for j, m in enumerate(meta):
        if m["kind"] == "cat":
            K = len(m["cats"])
            data.set_type_of_feature(j, K) #categorical with values in {0..K-1}
        else:
            data.set_type_of_feature(j, 0) #numeric
            mn, mx = m["bounds"]
            data.set_bounds_of_feature(j, mn, mx) #numeric bounds

    # add datapoints
    for i in range(n):
        data.add_data_point(X[i, :].tolist(), float(y[i]))

    return data


def set_points_per_tree(forest, n_train: int):
    """
    Ensure pyrfr has a positive number of datapoints per tree.
    Many builds require this explicitly (otherwise it can be 0 -> RuntimeError).
    """
    if not hasattr(forest, "options"):
        return forest

    opts = forest.options

    npt = max(2, int(n_train))

    for attr in ("num_data_points_per_tree", "n_points_per_tree"):
        if hasattr(opts, attr):
            setattr(opts, attr, npt)
            return forest
        
    raise RuntimeError(
        "Could not set num_data_points_per_tree / n_points_per_tree on this pyrfr build. "
        "Please print dir(forest.options) to see the correct attribute name."
    )

def make_rng(seed: int = 0):
    """
    pyrfr expects a RNG object (not an int) for fit(..., rng).
    Different builds expose different constructors; try common ones.
    """
    for name in ("default_random_engine", "random_engine", "rng"):
        if hasattr(reg, name):
            return getattr(reg, name)(seed)
    raise RuntimeError(
        "Could not find an RNG constructor in pyrfr.regression. "
        "Try inspecting dir(pyrfr.regression) to locate the RNG class for your build."
    )

def configure_fanova_forest(forest, params: dict):
    """
    Configure forest.options / forest.options.tree_opts robustly across pyrfr builds.
    """
    # forest-level
    if hasattr(forest, "options"):
        opts = forest.options
        if "num_trees" in params and hasattr(opts, "num_trees"):
            opts.num_trees = int(params["num_trees"])
        if "do_bootstrapping" in params and hasattr(opts, "do_bootstrapping"):
            opts.do_bootstrapping = bool(params["do_bootstrapping"])

        for k in ("num_data_points_per_tree", "n_points_per_tree"):
            if k in params and hasattr(opts, k):
                setattr(opts, k, int(params[k]))

        # tree opts
        if hasattr(opts, "tree_opts"):
            to = opts.tree_opts
            for key, attr in [
                ("max_depth", "max_depth"),
                ("max_features", "max_features"),
                ("min_samples_split", "min_samples_to_split"),
                ("min_samples_leaf", "min_samples_in_leaf"),
                ("epsilon_purity", "epsilon_purity"),
                ("max_num_nodes", "max_num_nodes"),
            ]:
                if key in params and hasattr(to, attr):
                    val = params[key]
                    # cast types
                    if attr in ("max_depth", "min_samples_to_split", "min_samples_in_leaf", "max_num_nodes", "max_features"):
                        val = int(val)
                    if attr == "epsilon_purity":
                        val = float(val)
                    setattr(to, attr, val)

    return forest