# Here you can find the test of the clean.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.corrupt import random_missing_values
from src.clean import mean_fill, do_nothing, knn_impute
import yaml 

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"


with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_drug = config["target_drug"]
corruption_rate = config["corruption_rate"]
columns = config["columns"]
seed = config["seed"]
drug_consumption = project_dir / f"data/processed/Drug_Consumption_{target_drug}.csv"

# load the corrupted dataframe and its mask
def load_corrupt_df():
    # read teh drug_consumption 
    df = pd.read_csv(drug_consumption)

    # corrupt the df using random_missing_values
    corrupt_df, mask_df = random_missing_values(df, corruption_rate, seed)

    return corrupt_df, mask_df

# load the original dataframe
def load_original_df():
    # read teh drug_consumption 
    df = pd.read_csv(drug_consumption)

    return df

def test_mean_fill_removes_missing_values():

    corrupt_df, mask = load_corrupt_df()

    cleaned_df = mean_fill(corrupt_df, columns, {})

    # one all for the columns -> True/ False second all for one value
    assert cleaned_df[columns["continuous"]].notna().all().all()

    print()
    print("The clean func does remove  NaNs")

def test_mean_fill(): 

    corrupt_df, mask_df = load_corrupt_df()
    
    cleaned_df = mean_fill(corrupt_df, columns, {})

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

    corrupted_df.loc[0, "Nscore"] = np.nan
    corrupted_df.loc[1, "Escore"] = np.nan
    corrupted_df.loc[2, "Oscore"] = np.nan

    cleaned_df = knn_impute(corrupted_df, columns, {"n_neighbors": 3})

    assert not cleaned_df.isna().any().any()

def test_knn_only_continuous():

    original_df = load_original_df() # get the original df
    corrupted_df, mask_df = random_missing_values(original_df, corruption_rate, seed)

    cleaned_df = knn_impute(corrupted_df, columns, {"n_neighbors": 3})

    continuous_missings = corrupted_df[columns["continuous"]].isna().sum().sum()
    total_missing = corrupted_df.isna().sum().sum()
    missings_after_cleanning = cleaned_df.isna().sum().sum()


    assert total_missing - missings_after_cleanning == continuous_missings