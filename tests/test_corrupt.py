# Here you can find the test of the corruption
# import pandas
from pathlib import Path

import pandas as pd
import yaml

from src.corrupt import (
    decimal_corruptor,
    duplicates_corruptor,
    label_correlated_missing,
    random_missing_values,
    self_masking_missing,
    typo_corruption,
)

# get the paths
project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

# load configuration file
with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

corruption_rate = config["corruption_rate"]
target_column = config["target_column"]
seed = config["seed"]
columns = config["columns"]

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

# load the original dataframe with continuous columns as float
# (the corruptors write float values, e.g. x / 10 or x + 0.00001, into int columns)
def load_float_df():
    df = load_original_df()
    df[columns["continuous"]] = df[columns["continuous"]].astype(float)

    return df



# test if corruption rate match the actual rate
def test_missing_rate():

    df, mask_df = load_corrupt_df()

    all_cells = df.shape[0] * len(columns["continuous"]) # get all the avaliable cells with the shape of the df
    actual_corruption_rate = mask_df.sum().sum() / all_cells # numbers of True cells over all the cells
    expected_corrupt_cells = round(all_cells * corruption_rate) # number of expected corrupted_cells
    corrupted_cells =  mask_df.sum().sum()


    # check if the target column contains NaNs
    assert mask_df[target_column].sum().sum() == 0, "The target column is corrupted!"

    assert corrupted_cells == expected_corrupt_cells, (
        f"Target corruption rate: {corruption_rate} \n Actual corruption rate: {actual_corruption_rate}"
    )

    print()
    print(f"Target corruption rate: {corruption_rate} \nActual corruption rate: {actual_corruption_rate}")

# test if the mask df actually reflects corrupted df and original df
def test_mask_df():

    # get the df and the mask_df
    corrupted_df, mask_df = load_corrupt_df()

    # get the original df
    original_df = load_original_df()


    for i in corrupted_df.index:
            for j in corrupted_df.columns:

                original_value = original_df.at[i, j]
                corrupted_value = corrupted_df.at[i, j]
                mask_value = mask_df.at[i, j]

                if mask_value:
                # If the mask says the cell was corrupted,
                # the corrupted value should be missing.
                    assert pd.isna(corrupted_value), (
                        f"Cell ({i}, {j}) is marked as corrupted "
                        "but is not missing"
                    )

                else:
                     # If the mask says the cell was NOT corrupted,
                    # its value should remain unchanged.
                    if pd.isna(original_value) and pd.isna(corrupted_value):
                        continue

                    assert corrupted_value == original_value, (
                        f"Cell ({i}, {j}) changed but the mask is False"
                    )

    print("The mask of the dataframe and corrupted dataframe do match")

# test if the original df was edited
def test_input_frame():

    original_df = load_original_df() # get the original df
    original_df_copy = original_df.copy() # copy the original

    #run it to corrupt
    random_missing_values(original_df, corruption_rate, seed)

    assert original_df.equals(original_df_copy)
    print("The input frame remained unchanged")

def test_mnar_rate():
    df = load_original_df()

    # corrupt the df using random_missing_values
    corrupt_df, mask_df = label_correlated_missing(df, corruption_rate, seed)

    eligible_cells = df.shape[0] * len(columns["continuous"]) # get all the avaliable cells with the shape of the df
    actual_corruption_rate = mask_df.sum().sum() / eligible_cells # numbers of True cells over all the cells
    expected_corrupt_cells = round(eligible_cells * corruption_rate) # number of expected corrupted_cells
    corrupted_cells =  mask_df[columns["continuous"]].sum().sum()


    # check if the target column contains NaNs
    assert mask_df[target_column].sum().sum() == 0, "The target column is corrupted!"

    assert expected_corrupt_cells == corrupted_cells, (
        f"Target corruption rate: {corruption_rate} \n Actual corruption rate: {actual_corruption_rate}"
    )

    print()
    print(f"Target corruption rate: {corruption_rate} \nActual corruption rate: {actual_corruption_rate}")


def test_typos_rate():
    df = load_original_df()

    # corrupt the df using random_missing_values
    corrupt_df, mask_df = typo_corruption(df, corruption_rate, seed)

    eligible_columns = columns["typo_columns"]
    eligible_cells = df.shape[0] * len(eligible_columns) # get all the avaliable cells with the shape of the df
    actual_corruption_rate = mask_df.sum().sum() / eligible_cells # numbers of True cells over all the cells
    expected_corrupt_cells = round(eligible_cells * corruption_rate) # number of expected corrupted_cells
    corrupted_cells =  mask_df[eligible_columns].sum().sum()


    # check if the target column contains typos
    assert mask_df[target_column].sum().sum() == 0, "The target column is corrupted!"

    assert expected_corrupt_cells == corrupted_cells, (
        f"Target corruption rate: {corruption_rate} \n Actual corruption rate: {actual_corruption_rate}"
    )

    print()
    print(f"Target corruption rate: {corruption_rate} \nActual corruption rate: {actual_corruption_rate}")

def test_typo_columns():

    # read teh csv
    df = load_original_df()

    corrupt_df, mask_df = typo_corruption(df, corruption_rate, seed)
    corrupted_columns = []

    eligible_columns = columns["typo_columns"]

    for i in columns["categorical"]:
        if i not in eligible_columns:
            corrupted_columns.append(i)

    sum_all_other_columns = mask_df[corrupted_columns].sum().sum().sum()
    sum_typo_education = mask_df[eligible_columns[0]].sum().sum()
    sum_type_country = mask_df[eligible_columns[1]].sum().sum()

    assert sum_all_other_columns == 0 and sum_typo_education != 0 and sum_type_country != 0

def test_decimal_corruptor():

    df = load_float_df()

    # only non zero cells in the decimal columns can be corrupted

    all_cells = (df[columns["decimal_columns"]] != 0).sum().sum()

    df, mask_df = decimal_corruptor(df, corruption_rate, seed)

    actual_corruption_rate = mask_df.sum().sum() / all_cells # numbers of True cells over all the cells
    expected_corrupt_cells = round(all_cells * corruption_rate) # number of expected corrupted_cells
    corrupted_cells =  mask_df.sum().sum()


    # check if the target column contains NaNs
    assert mask_df[target_column].sum().sum() == 0, "The target column is corrupted!"

    assert corrupted_cells == expected_corrupt_cells, (
        f"Target corruption rate: {corruption_rate} \n Actual corruption rate: {actual_corruption_rate}"
    )

    print()
    print(f"Target corruption rate: {corruption_rate} \nActual corruption rate: {actual_corruption_rate}")

def test_duplicate_row_count():

    df = load_float_df()

    corrupt_df, mask_df = duplicates_corruptor(df, corruption_rate, seed)

    expected_row_count = round(df.shape[0] * corruption_rate)
    extra_rows = len(corrupt_df) - len(df)
    actual_row_count = len(mask_df)

    assert expected_row_count == actual_row_count and extra_rows == actual_row_count

# test if the original df was edited
def test_input_frame_duplicate():

    original_df = load_float_df() # get the original df
    original_df_copy = original_df.copy() # copy the original

    #run it to corrupt
    duplicates_corruptor(original_df, corruption_rate, seed)

    assert original_df.equals(original_df_copy)
    print("The input frame remained unchanged")

def test_duplicate_id():

    original_df = load_float_df() # get the original df

    corrupted_df, mask_df = duplicates_corruptor(original_df, corruption_rate, seed)

    # get the i of the mask df
    new_ids = mask_df["new_row_id"]

    # get the id in the original_df
    original_ids = original_df.index

    assert (~new_ids.isin(original_ids)).all()


# random missing values should only touch the continuous columns
def test_missing_only_continuous():

    corrupt_df, mask_df = load_corrupt_df()

    other_columns = [c for c in mask_df.columns if c not in columns["continuous"]]

    assert mask_df[other_columns].sum().sum() == 0, "A non continuous column was corrupted"

def test_self_masking_rate():

    df = load_original_df()

    corrupt_df, mask_df = self_masking_missing(df, corruption_rate, seed)

    # every continuous column loses the same number of cells
    expected_per_column = round(df.shape[0] * corruption_rate)

    for j in columns["continuous"]:
        assert mask_df[j].sum() == expected_per_column
        assert corrupt_df.loc[mask_df[j], j].isna().all()

    other_columns = [c for c in mask_df.columns if c not in columns["continuous"]]
    assert mask_df[other_columns].sum().sum() == 0, "A non continuous column was corrupted"

# self masking should remove the high values more often than the low ones
def test_self_masking_hits_high_values():

    df = load_original_df()

    corrupt_df, mask_df = self_masking_missing(df, corruption_rate, seed)

    for j in ["lead_time", "adr"]:
        missing_mean = df.loc[mask_df[j], j].mean()
        observed_mean = df.loc[~mask_df[j], j].mean()

        assert missing_mean > observed_mean, f"{j}: missing values are not higher than the observed ones"

def test_self_masking_input_frame():

    original_df = load_original_df() # get the original df
    original_df_copy = original_df.copy() # copy the original

    self_masking_missing(original_df, corruption_rate, seed)

    assert original_df.equals(original_df_copy)
