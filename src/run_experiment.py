import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from src.corrupt import random_missing_values, mnar_missing_values, typo_corruption
from src.clean import mean_fill, do_nothing, knn_impute, iterative_imputation, fuzzy_match, drop_rows
from src.evaluate import evaluate
from scipy.stats import wasserstein_distance
import yaml

from src.load import load_model_data 

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"
results_path = project_dir / "results/experiments_partial.csv"
raw_csv = project_dir / f"data/raw/hotel_bookings.csv"



with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_column = config["target_column"]
columns = config["columns"]
experiment = config["experiment"]
seeds = experiment["seeds"]
corruption_rates = experiment["corruption_rates"]
strategies = experiment["strategies"]
corruption_types = experiment["corruption_types"]
models = experiment["models"]
strategy_params = config["strategy_params"]

strategies_map = {
    "mean_fill": mean_fill,
    "do_nothing": do_nothing,
    "knn_impute": knn_impute,
    "iterative_imputation": iterative_imputation,
    "fuzzy_match": fuzzy_match,
    "drop_rows": drop_rows
}

corruption_types_map = {
    "random_missing_values": random_missing_values,
    "mnar_missing_values": mnar_missing_values,
    "typo_corruption": typo_corruption,
}

def run_experiment(df): 

    original_df = df.copy()
    original_df = original_df.sample(n=20000, random_state=42)

    if results_path.exists():
        results_path.unlink()

    for seed in seeds:
        train_df, holdout_df = train_test_split(
            original_df, test_size=0.2, random_state=seed, stratify=original_df[target_column]
        )

        seed_rows = []

        for corruption_rate in corruption_rates:
            for corruption_type in corruption_types:
                corrupt_func = corruption_types_map[corruption_type]
                # corrupt the df with certain type
                corrupted_df, mask_df = corrupt_func(train_df, corruption_rate, seed)
                for strategy in strategies:
                    # get the functions from map
                    clean_func = strategies_map[strategy]

                    # clean the df with certain strategy
                    cleaned_df = clean_func(corrupted_df, columns, strategy_params[strategy])
                    for model in models:
                
                        # evaluate the cleaned df
                        metrics = metrics = evaluate(
                            train_df=train_df,
                            dirty_df=corrupted_df,
                            cleaned_df=cleaned_df,
                            mask=mask_df,
                            corruption_type=corruption_type,
                            cols=columns,
                            target_column=target_column,
                            holdout_df=holdout_df,
                            model_name=model,
                            seed=seed,
                        )
                        
                        row = {
                            "model": model,
                            "seed": seed,
                            "corruption_rate": corruption_rate,
                            "strategy": strategy,
                            "corruption_type": corruption_type,
                            **metrics,
                        }

                        seed_rows.append(row)

        header_needed = not results_path.exists()
        pd.DataFrame(seed_rows).to_csv(results_path, mode="a", header=header_needed, index=False)
        print(f"seed {seed} done", flush=True)

    result_df = pd.read_csv(results_path)
    result_df.to_parquet(project_dir / "results/experiments.parquet")
    return result_df

if __name__ == "__main__":
    original_df = load_model_data(raw_csv)

    run_experiment(original_df)

    

