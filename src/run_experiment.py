# Here you can find the test of the clean.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.corrupt import random_missing_values, mnar_missing_values, typo_corruption
from src.clean import mean_fill, do_nothing, knn_impute, iterative_imputation
from src.evaluate import evaluate_continuous_recovery
from scipy.stats import wasserstein_distance
import yaml 

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"


with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_drug = config["target_drug"]
columns = config["columns"]
experiment = config["experiment"]
seeds = experiment["seeds"]
corruption_rates = experiment["corruption_rates"]
strategies = experiment["strategies"]
corruption_types = experiment["corruption_types"]
drug_consumption = project_dir / f"data/processed/Drug_Consumption_{target_drug}.csv"

strategies_map = {
    "mean_fill": mean_fill,
    "do_nothing": do_nothing,
    "knn_impute": knn_impute,
    "iterative_imputation": iterative_imputation,
}

corruption_types_map = {
    "random_missing_values": random_missing_values,
    "mnar_missing_values": mnar_missing_values,
    "typo_corruption": typo_corruption,
}

def run_experiment(df): 

    original_df = df.copy()

    results = []

    for seed in seeds:
        for corruption_rate in corruption_rates:
            for corruption_type in corruption_types:
                corrupt_func = corruption_types_map[corruption_type]
                # corrupt the df with certain type
                corrupted_df, mask_df = corrupt_func(original_df, corruption_rate, seed)
                for strategy in strategies:
                
                    # get the functions from map
                    clean_func = strategies_map[strategy]

                    # clean the df with certain strategy
                    cleaned_df = clean_func(corrupted_df, columns, {"n_neighbors": 3, "max_iter": 10})

                    # evaluate the cleaned df
                    evaluated_metrics = evaluate_continuous_recovery(original_df, cleaned_df, mask_df, columns)

                    # create an dictionary with values
                    row = {
                        "seed": seed,
                        "corruption_rate": corruption_rate,
                        "strategy": strategy,
                        "corruption_type": corruption_type,
                        "recovery_applicable": evaluated_metrics["recovery_applicable"],
                        "rmse": evaluated_metrics["rmse"],
                        "mae": evaluated_metrics["mae"],
                        "bias": evaluated_metrics["bias"],
                        "variance_ratio": evaluated_metrics["variance_ratio"],
                        "wasserstein": evaluated_metrics["wasserstein"]
                    }

                    results.append(row)

    # create a result df
    result_df = pd.DataFrame(results)

    # save it as parquet
    file_path = project_dir / "results/experiments.parquet"
    result_df.to_parquet(file_path)

    return result_df
    

