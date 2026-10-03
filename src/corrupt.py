# Here you can find the functions that corrupt the data set

import pandas as pd # import pandas
import random # import random for corrupting 
from pathlib import Path
import yaml
import numpy as np
import string

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

# load configuration file
with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_drug = config["target_drug"]

avaliable_columns = config["columns"]["continuous"] + config["columns"]["categorical"]
continuous_columns = config["columns"]["continuous"]

# random missing corruptor
def random_missing_values(df, corruption_rate, seed):

    all_cells = df.shape[0] * (df.shape[1]) # availiable cells (without target one)
    corrupted_cells = round(all_cells * corruption_rate) #number of cells that has to be corrupted 

    corrupted_df = df.copy() # creaet a copy of df to corrupt
    mask_df = pd.DataFrame(False, index=df.index, columns=avaliable_columns+[target_drug]) # create a df to fill with Fase (non empty cells)
    sample_list = [] # creating all possible pairs based on rows and columns

    # get all the possible pearis row + column
    for i in df.index:
        for j in avaliable_columns:
            sample_list.append((i, j))

    rng = random.Random(seed)

    # simple random sample ne = corrupted_cells
    random_sample_list = rng.sample(sample_list, corrupted_cells)

    # corrupt cells
    for i in random_sample_list:
        corrupted_df.at[i[0], i[1]] = None
        mask_df.at[i[0], i[1]] = True

    return corrupted_df, mask_df

# mnar corruptor 
def mnar_missing_values(df, corruption_rate, seed): 

    corrupted_df = df.copy() # creaet a copy of df to corrupt

    eligible_cells = df.shape[0] * len(continuous_columns)

    mask_df = pd.DataFrame(False, index=df.index, columns=avaliable_columns+[target_drug]) # create a df to fill with Fase (non empty cells)

    high_usage_sample = []
    low_usage_sample = []

    # get all the possible pearis row + column
    for i in df.index:
        for j in continuous_columns:
            if corrupted_df.at[i, target_drug] == 1: 
                high_usage_sample.append((i, j))
            else:
                low_usage_sample.append((i, j))

    rng = random.Random(seed)

    high_usage_proportion = corrupted_df[target_drug].sum() / df.shape[0]
    low_usage_proportion = (df.shape[0] - corrupted_df[target_drug].sum()) / df.shape[0]

    A = np.array([
        [high_usage_proportion, low_usage_proportion],
        [1, -3]
    ])

    b = np.array([corruption_rate, 0])

    solution = np.linalg.solve(A, b)

    expected_corrupt_cells = round(eligible_cells * corruption_rate)

    high_usage_cells = round(solution[0] * len(high_usage_sample))
    low_usage_cells = expected_corrupt_cells - high_usage_cells

    # simple random sample ne = corrupted_cells
    random_sample_list_high_usage = rng.sample(high_usage_sample, high_usage_cells)
    random_sample_list_low_usage = rng.sample(low_usage_sample, low_usage_cells)

    for i in random_sample_list_high_usage:
        corrupted_df.at[i[0], i[1]] = None
        mask_df.at[i[0], i[1]] = True

    for i in random_sample_list_low_usage:
        corrupted_df.at[i[0], i[1]] = None
        mask_df.at[i[0], i[1]] = True

    return corrupted_df, mask_df


def typo_corruption(df, corruption_rate, seed):

    corrupted_df = df.copy() # creaet a copy of df to corrupt
    eligible_columns = ["Education", "Country"]
    eligible_cells = corrupted_df.shape[0] * len(eligible_columns)
    corrupted_cells = round(eligible_cells * corruption_rate)

    mask_df = pd.DataFrame(False, index=df.index, columns=df.columns) # create a df to fill with Fase (non empty cells)

    sample_list = [] # creating all possible pairs based on rows and columns
    
    # get all the possible pearis row + column
    for i in df.index:
        for j in eligible_columns:
            sample_list.append((i, j))

    rng = random.Random(seed)

    random_sample_list = rng.sample(sample_list, corrupted_cells)

    for i in random_sample_list:
        original_word = corrupted_df.at[i[0], i[1]]
        original_word_array = list(corrupted_df.at[i[0], i[1]])
        position = rng.randrange(len(original_word))
        replacement = rng.choice(string.ascii_letters)

        while replacement == original_word_array[position]:
            replacement = rng.choice(string.ascii_letters)

        original_word_array[position] = replacement

        new_word = "".join(original_word_array)

        corrupted_df.at[i[0], i[1]] = new_word
        mask_df.at[i[0], i[1]] = True

    return corrupted_df, mask_df



