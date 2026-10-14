# Here you can find the test of the clean.py
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.clean import do_nothing, fuzzy_match, knn_impute, mean_fill, mean_fill_indicator
from src.corrupt import random_missing_values, typo_corruption

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"


with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_column = config["target_column"]
corruption_rate = config["corruption_rate"]
columns = config["columns"]
seed = config["seed"]
# small committed sample of the processed data, so the tests also run in CI
csv = project_dir / "tests/data/hotel_bookings_sample.csv"
sample_size = 2000 # sample the rows so the tests run fast

# load the corrupted dataframe and its mask
def load_corrupt_df():
    # read teh csv
    df = pd.read_csv(csv).sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # corrupt the df using random_missing_values
    corrupt_df, mask_df = random_missing_values(df, corruption_rate, seed)

    return corrupt_df, mask_df

# load the original dataframe
def load_original_df():
    # read teh csv
    df = pd.read_csv(csv).sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df

def test_mean_fill_removes_missing_values():

    corrupt_df, mask = load_corrupt_df()

    cleaned_df, _ = mean_fill(corrupt_df, columns, {})

    # one all for the columns -> True/ False second all for one value
    assert cleaned_df[columns["continuous"]].notna().all().all()

    print()
    print("The clean func does remove  NaNs")

def test_mean_fill():

    corrupt_df, mask_df = load_corrupt_df()

    cleaned_df, _ = mean_fill(corrupt_df, columns, {})

    # hash map with all the means for each column
    means = {}

    # calculate mean for each value
    for i in columns["continuous"]:
        means[i] = corrupt_df[i].mean()

    # got through each cell that is missing based on masked_df and chek if the cleand_df filled it with mean
    for i in corrupt_df.index:
        for j in columns["continuous"]:
            if mask_df.at[i, j]:
                assert cleaned_df.at[i, j] == means[j], "The missing value was not filled with the mean of that column"

    print("The cells were successfully filled with mean values!")

# test if the original df was edited
def test_input_frame():

    original_df = load_original_df() # get the original df
    original_df_copy = original_df.copy() # copy the original

    #run it to corrupt
    mean_fill(original_df, columns, {})

    assert original_df.equals(original_df_copy)
    print("The input frame remained unchanged")

def test_do_nothing_input_frame():

    original_df = load_original_df() # get the original df
    original_df_copy = original_df.copy() # copy the original

    #run it to corrupt
    do_nothing(original_df, columns, {})

    assert original_df.equals(original_df_copy)
    print("The input frame remained unchanged")


def test_knn_removes_missing_values():

    original_df = load_original_df() # get the original df

    corrupted_df = original_df.copy()

    corrupted_df.loc[0, "lead_time"] = np.nan
    corrupted_df.loc[1, "adr"] = np.nan
    corrupted_df.loc[2, "adults"] = np.nan

    cleaned_df, _ = knn_impute(corrupted_df, columns, {"n_neighbors": 3})

    # only the continuous columns are imputed (agent has NaNs in the raw data)
    assert not cleaned_df[columns["continuous"]].isna().any().any()

def test_knn_only_continuous():

    original_df = load_original_df() # get the original df
    corrupted_df, mask_df = random_missing_values(original_df, corruption_rate, seed)

    cleaned_df, _ = knn_impute(corrupted_df, columns, {"n_neighbors": 3})

    continuous_missings = corrupted_df[columns["continuous"]].isna().sum().sum()
    total_missing = corrupted_df.isna().sum().sum()
    missings_after_cleanning = cleaned_df.isna().sum().sum()


    assert total_missing - missings_after_cleanning == continuous_missings


# the holdout has to be filled with the train means, not its own
def test_mean_fill_holdout_uses_train_means():

    corrupt_df, mask_df = load_corrupt_df()
    holdout_df = corrupt_df.copy()
    holdout_df.loc[0, "lead_time"] = np.nan
    holdout_df.loc[:, "adr"] = np.nan

    cleaned_df, cleaned_holdout = mean_fill(corrupt_df, columns, {}, holdout_df)

    assert cleaned_holdout.at[0, "lead_time"] == corrupt_df["lead_time"].mean()
    assert (cleaned_holdout["adr"] == corrupt_df["adr"].mean()).all()

def test_mean_fill_indicator_flags():

    corrupt_df, mask_df = load_corrupt_df()

    cleaned_df, _ = mean_fill_indicator(corrupt_df, columns, {})

    for j in columns["continuous"]:
        # the flag is 1 exactly where the value was corrupted
        assert (cleaned_df[f"{j}_missing"] == mask_df[j].astype(int)).all()

    assert cleaned_df[columns["continuous"]].notna().all().all()

# a clean holdout still gets the flag columns (all zeros) so it matches the train columns
def test_mean_fill_indicator_holdout_columns():

    corrupt_df, mask_df = load_corrupt_df()
    holdout_df = load_original_df()

    cleaned_df, cleaned_holdout = mean_fill_indicator(corrupt_df, columns, {}, holdout_df)

    assert list(cleaned_df.columns) == list(cleaned_holdout.columns)
    flag_columns = [f"{j}_missing" for j in columns["continuous"]]
    assert cleaned_holdout[flag_columns].sum().sum() == 0


# the holdout typos are mapped with the categories learned on train
def test_fuzzy_match_holdout():

    original_df = load_original_df()
    corrupt_df, _ = typo_corruption(original_df, corruption_rate, seed)

    holdout_df = original_df.copy()
    holdout_df.loc[0, "market_segment"] = "Onlihe TA"   # one letter typo of a long category
    holdout_df.loc[1, "customer_type"] = "Bogus-Category-Name"   # nothing close in train

    cleaned_df, cleaned_holdout = fuzzy_match(corrupt_df, columns, {}, holdout_df)

    assert cleaned_holdout.at[0, "market_segment"] == "Online TA"
    assert cleaned_holdout.at[1, "customer_type"] == "Bogus-Category-Name"

# a case-only typo that appears first must not rename the whole category
def test_fuzzy_match_keeps_most_frequent_spelling():

    original_df = load_original_df()
    corrupt_df = original_df.copy()
    corrupt_df.loc[0, "arrival_date_month"] = corrupt_df.loc[0, "arrival_date_month"].upper()

    cleaned_df, _ = fuzzy_match(corrupt_df, columns, {})

    assert (cleaned_df["arrival_date_month"] == original_df["arrival_date_month"]).all()
