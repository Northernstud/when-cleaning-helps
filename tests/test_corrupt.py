# Here you can find the test of the corruption
# import pandas
import pandas as pd 
from pathlib import Path
from src.corrupt import random_missing_values
import yaml

# get the paths
project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

# load configuration file
with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

corruption_rate = config["corruption_rate"]
target_drug = config["target_drug"]
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
     
# test if corruption rate match the actual rate
def test_missing_rate():

    df, mask_df = load_corrupt_df()

    all_cells = df.shape[0] * (df.shape[1]) # get all the avaliable cells with the shape of the df
    actual_corruption_rate = mask_df.sum().sum() / all_cells # numbers of True cells over all the cells
    expected_corrupt_cells = round(all_cells * corruption_rate) # number of expected corrupted_cells
    corrupted_cells =  mask_df.sum().sum()


    # check if the target column contains NaNs
    if mask_df[target_drug].sum().sum() != 0:
        assert False, "The target column is corrupted!"

    assert corrupted_cells == expected_corrupt_cells, f"Target corruption rate: {corruption_rate} \n Actual corruption rate: {actual_corruption_rate}"

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

