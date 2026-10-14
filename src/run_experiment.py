from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.clean import do_nothing, drop_rows, fuzzy_match, knn_impute, mean_fill, mean_fill_indicator
from src.corrupt import label_correlated_missing, random_missing_values, self_masking_missing, typo_corruption
from src.evaluate import evaluate
from src.load import load_model_data

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"
results_path = project_dir / "results/experiments_partial.csv"
parquet_path = project_dir / "results/experiments.parquet"
raw_csv = project_dir / "data/raw/hotel_bookings.csv"



with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_column = config["target_column"]
columns = config["columns"]
experiment = config["experiment"]
seeds = experiment["seeds"]
corruption_rates = experiment["corruption_rates"]
strategies = experiment["strategies"]
corruption_types = experiment["corruption_types"]
holdout_conditions = experiment["holdout_conditions"]
models = experiment["models"]
strategy_params = config["strategy_params"]

strategies_map = {
    "mean_fill": mean_fill,
    "mean_fill_indicator": mean_fill_indicator,
    "do_nothing": do_nothing,
    "knn_impute": knn_impute,
    "fuzzy_match": fuzzy_match,
    "drop_rows": drop_rows
}

corruption_types_map = {
    "random_missing_values": random_missing_values,
    "label_correlated_missing": label_correlated_missing,
    "self_masking_missing": self_masking_missing,
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

                # the deployment data gets the same corruption (different seed so the pattern is independent)
                corrupted_holdout, _ = corrupt_func(holdout_df, corruption_rate, seed + 1000)

                # stack both holdout versions so they are cleaned in one pass with the same fitted cleaner
                holdouts = pd.concat(
                    {"clean": holdout_df, "corrupted": corrupted_holdout}
                )
                holdouts = holdouts.loc[holdout_conditions]

                for strategy in strategies:
                    # get the functions from map
                    clean_func = strategies_map[strategy]

                    # clean the df with certain strategy, the holdouts are cleaned with what was learned on train
                    cleaned_df, cleaned_holdouts = clean_func(
                        corrupted_df, columns, strategy_params[strategy], holdouts
                    )

                    for holdout_condition in holdout_conditions:
                        eval_holdout = cleaned_holdouts.loc[holdout_condition]

                        for model in models:

                            # evaluate the cleaned df
                            metrics = evaluate(
                                train_df=train_df,
                                dirty_df=corrupted_df,
                                cleaned_df=cleaned_df,
                                mask=mask_df,
                                corruption_type=corruption_type,
                                cols=columns,
                                target_column=target_column,
                                holdout_df=eval_holdout,
                                model_name=model,
                                seed=seed,
                            )

                            row = {
                                "model": model,
                                "seed": seed,
                                "corruption_rate": corruption_rate,
                                "strategy": strategy,
                                "corruption_type": corruption_type,
                                "holdout_condition": holdout_condition,
                                **metrics,
                            }

                            seed_rows.append(row)

        header_needed = not results_path.exists()
        pd.DataFrame(seed_rows).to_csv(results_path, mode="a", header=header_needed, index=False)
        print(f"seed {seed} done", flush=True)

    result_df = pd.read_csv(results_path)
    result_df.to_parquet(parquet_path)
    return result_df

if __name__ == "__main__":
    original_df = load_model_data(raw_csv)

    run_experiment(original_df)


