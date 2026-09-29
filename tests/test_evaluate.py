# Here you can find the test of the clean.py
import pandas as pd
from pathlib import Path
from src.corrupt import random_missing_values
from src.clean import mean_fill
from src.evaluate import continuous_recovery
import yaml 

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"


with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_drug = config["target_drug"]
corruption_rate = config["corruption_rate"]
columns = config["columns"]
drug_consumption = project_dir / f"data/processed/Drug_Consumption_{target_drug}.csv"

def load_corrupt_df():
    # read teh drug_consumption 
    df = pd.read_csv(drug_consumption)

    # corrupt the df using random_missing_values
    corrupt_df, mask_df = random_missing_values(df, corruption_rate)

    return corrupt_df, mask_df

# load the original dataframe
def load_original_df():
    # read teh drug_consumption 
    df = pd.read_csv(drug_consumption)

    return df

def test_continuous_recovery_returns_metrics():

    cols = columns["continuous"]
    corrupted_df, mask_df = load_corrupt_df()
    cleaned_df = mean_fill(corrupted_df, cols, {})
    original_df = load_original_df()
    

    stats = continuous_recovery(original_df, cleaned_df, mask_df, cols)

    metrics = [
        "rmse",
        "mae",
        "bias",
        "variance_ratio",
        "wasserstein"
    ]

    for key, value in stats:
        if key not in metrics:
            assert False, f"The key '{key}' is not an expected metrics"
        




