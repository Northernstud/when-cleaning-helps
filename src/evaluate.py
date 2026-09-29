# Here you can find functions that evaluate cleaning methods
import pandas as pd 
import numpy as np 
import yaml
from pathlib import Path
from scipy.stats import wasserstein_distance

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_drug = config["target_drug"]

def evaluate_continuous_recovery(original_df, cleaned_df, mask_df, cols):
    
    # create a copy of dfs 
    original_df = original_df.copy()
    cleaned_df = cleaned_df.copy()
    mask_df = mask_df.copy()

    cols = cols["continuous"]

    # data from original and from cleaned df
    true_values = []
    imputed_values = []

    for i in mask_df.index:
        for j in cols:
            if mask_df.at[i, j]:
                true_values.append(original_df.at[i, j])
                imputed_values.append(cleaned_df.at[i, j])

    residuals = np.array(true_values) - np.array(imputed_values)

    # calculate rmse
    rmse = np.sqrt(np.mean(residuals ** 2))

    # claculate mae
    mae = np.mean(np.abs(residuals))

    # calculate bias
    bias = np.mean(residuals)

    # calculate variance ratio 
    if len(imputed_values) < 2:
        variance_ratio = None
    else:
        variance_ratio = np.var(np.array(imputed_values)) / np.var(np.array(true_values))

    # Wasserstein distance
    wasserstein = wasserstein_distance(imputed_values, true_values)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "variance_ratio": variance_ratio,
        "wasserstein": wasserstein
    }