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
corruption_families = config["corruption_families"]

NOT_APPLICABLE = {
        "rmse": None,
        "mae": None,
        "bias": None,
        "variance_ratio": None,
        "wasserstein": None,
        "recovery_applicable": False
    }  

CAT_NOT_APPLICABLE = {
        "cat_match_rate": None,
        "cat_unresolved_rate": None,
        "cat_wrong_label_rate": None,
        "cat_recovery_applicable": None,
    }

def evaluate(original_df, dirty_df, cleaned_df, mask, corruption_type, cols):
    out = empty_metric_template()
    family = corruption_families[corruption_type]
    if family == "continuous":
        out.update(evaluate_continuous_recovery(original_df, cleaned_df, mask, cols))
    elif family == "categorical":
        out.update(evaluate_categorical_recovery(original_df, cleaned_df, mask, dirty_df, cols))
    # elif family == "row":
    #     out.update(evaluate_detection_quality(mask, cleaned_df))
    return out

def empty_metric_template():
    return {
        "rmse": None,
        "mae": None,
        "bias": None,
        "variance_ratio": None,
        "wasserstein": None,
        "recovery_applicable": None,
        "cat_match_rate": None,
        "cat_nan_rate": None,
        "cat_untouched_rate": None,
        "cat_wrong_label_rate": None,
        "cat_recovery_applicable": None,
        "detect_precision": None,
        "detect_recall": None,
        "detect_f1": None,
        "detect_applicable": None,
    }

def evaluate_continuous_recovery(original_df, cleaned_df, mask_df, cols):
    
    # create a copy of dfs 
    original_df = original_df.copy()
    cleaned_df = cleaned_df.copy()
    mask_df = mask_df.copy()

    cols = cols["continuous"]

    # data from original and from cleaned df

    recovery_applicable = True

    common = mask_df.index.intersection(cleaned_df.index)
    m = mask_df.loc[common, cols].to_numpy()

    if m.sum() == 0:
        return NOT_APPLICABLE

    true_values = original_df.loc[common, cols].to_numpy()[m]
    imputed_values = cleaned_df.loc[common, cols].to_numpy()[m]

    if np.isnan(imputed_values).any():
        return NOT_APPLICABLE
    
    residuals = true_values - imputed_values

    # calculate rmse
    rmse = np.sqrt(np.mean(residuals ** 2))

    # claculate mae
    mae = np.mean(np.abs(residuals))

    # calculate bias
    bias = np.mean(residuals)

    # calculate variance ratio 
    if len(imputed_values) < 2 or np.var(true_values) == 0:
        variance_ratio = None
    else:
        variance_ratio = np.var(imputed_values) / np.var(true_values)

    # Wasserstein distance
    wasserstein = wasserstein_distance(imputed_values, true_values)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "variance_ratio": variance_ratio,
        "wasserstein": wasserstein,
        "recovery_applicable": recovery_applicable
    }

def evaluate_categorical_recovery(original_df, cleaned_df, mask_df, dirty_df, cols):
     # create a copy of dfs 
    original_df = original_df.copy()
    cleaned_df = cleaned_df.copy()
    mask_df = mask_df.copy()

    cols = ["Education", "Country"]

    common = mask_df.index.intersection(cleaned_df.index)
    m = mask_df.loc[common, cols].to_numpy()


    if m.sum() == 0:
        return CAT_NOT_APPLICABLE

    
    true_values = original_df.loc[common, cols].to_numpy()[m]
    imputed_values = cleaned_df.loc[common, cols].to_numpy()[m]
    nans = pd.isna(imputed_values).sum()

    total_corrupted = m.sum()

    matches = (true_values == imputed_values).sum()
    dirty_values = dirty_df.loc[common, cols].to_numpy()[m]
    non_matches = total_corrupted - matches
    non_nan = ~pd.isna(imputed_values)
    untouched = ((imputed_values == dirty_values) & non_nan & (true_values != imputed_values)).sum()

    wrong_label = total_corrupted - matches - nans - untouched

    cat_recovery_applicable = True



    # match rate
    cat_match_rate      = float(matches / total_corrupted)
    cat_nan_rate        = float(nans / total_corrupted)
    cat_untouched_rate  = float(untouched / total_corrupted)
    cat_wrong_label_rate = float(wrong_label / total_corrupted)

    return {
        "cat_match_rate": cat_match_rate,
        "cat_untouched_rate": cat_untouched_rate,
        "cat_wrong_label_rate": cat_wrong_label_rate,
        'cat_nan_rate': cat_nan_rate,
        "cat_recovery_applicable": cat_recovery_applicable,
    }
