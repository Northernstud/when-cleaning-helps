# Here you can find the test of the evaluate.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.corrupt import random_missing_values
from src.clean import mean_fill
from src.evaluate import evaluate_continuous_recovery
from scipy.stats import wasserstein_distance
import yaml 

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"


with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_column = config["target_column"]
corruption_rate = config["corruption_rate"]
columns = config["columns"]
seed = config["seed"]
csv = project_dir / f"data/processed/csv_{target_column}.csv"


def load_corrupt_df():
    # read teh csv 
    df = pd.read_csv(csv)

    # corrupt the df using random_missing_values
    corrupt_df, mask_df = random_missing_values(df, corruption_rate, seed)

    return corrupt_df, mask_df

# load the original dataframe
def load_original_df():
    # read teh csv 
    df = pd.read_csv(csv)

    return df

def test_evaluate_continuous_recovery_returns_metrics():

    corrupted_df, mask_df = load_corrupt_df()
    cleaned_df = mean_fill(corrupted_df, columns, {})
    original_df = load_original_df()
    

    stats = evaluate_continuous_recovery(original_df, cleaned_df, mask_df, columns)

    metrics = [
        "rmse",
        "mae",
        "bias",
        "variance_ratio",
        "wasserstein",
        "recovery_applicable"
    ]

    assert set(stats.keys()) == set(metrics), f"The keys do not match the requirements"

    print()
    print("All the keys match the required keys")

def test_metrics_accuracy():

    original_data = {
        "Nscore": [1.0, 2.0, 3.0, 4.0],
    }

    cleaned_data = {
        "Nscore": [1.0, 1.5, 3.0, 1.5],
    }

    mask_data = {
        "Nscore": [False, True, False, True],
    }

    cols = {
        "continuous": ["Nscore"]
    }

    original_df = pd.DataFrame(original_data)
    cleaned_df = pd.DataFrame(cleaned_data)
    mask_df= pd.DataFrame(mask_data)

    true_values = []
    imputed_values = []

    for i in mask_df.index:
            for j in cols["continuous"]:
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

    true_metrics = {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "variance_ratio": variance_ratio,
        "wasserstein": wasserstein,
        "recovery_applicable": True
    }

    evaluated_metrics = evaluate_continuous_recovery(original_df, cleaned_df, mask_df, cols)

    assert true_metrics == evaluated_metrics, "Evaluation function does not return accurate metrics"

        




