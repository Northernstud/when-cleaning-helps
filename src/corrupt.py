# Here you can find the functions that corrupt the data set

import random  # import random for corrupting
import string
from pathlib import Path

import numpy as np
import pandas as pd  # import pandas
import yaml

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

# load configuration file
with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_column = config["target_column"]

avaliable_columns = config["columns"]["continuous"] + config["columns"]["categorical"]
continuous_columns = config["columns"]["continuous"]
decimal_columns = config["columns"]["decimal_columns"]
typo_columns = config["columns"]["typo_columns"]
self_masking_strength = config["corruption_params"]["self_masking_missing"]["strength"]

# random missing corruptor
def random_missing_values(df, corruption_rate, seed):

    all_cells = df.shape[0] * len(continuous_columns) # availiable cells (without target one)
    corrupted_cells = round(all_cells * corruption_rate) #number of cells that has to be corrupted

    corrupted_df = df.copy() # creaet a copy of df to corrupt
    # create a df to fill with Fase (non empty cells)
    mask_df = pd.DataFrame(False, index=df.index, columns=avaliable_columns+[target_column])
    sample_list = [] # creating all possible pairs based on rows and columns

    # get all the possible pearis row + column
    # only continuous columns, same cells as label_correlated_missing so the mechanisms are comparable
    for i in df.index:
        for j in continuous_columns:
            sample_list.append((i, j))

    rng = random.Random(seed)

    # simple random sample ne = corrupted_cells
    random_sample_list = rng.sample(sample_list, corrupted_cells)

    # corrupt cells
    for i in random_sample_list:
        corrupted_df.at[i[0], i[1]] = None
        mask_df.at[i[0], i[1]] = True

    return corrupted_df, mask_df

# self-masking (true mnar) corruptor: the higher the value, the more likely it goes missing
def self_masking_missing(df, corruption_rate, seed):

    corrupted_df = df.copy() # creaet a copy of df to corrupt
    # create a df to fill with Fase (non empty cells)
    mask_df = pd.DataFrame(False, index=df.index, columns=avaliable_columns+[target_column])

    rng = np.random.default_rng(seed)

    # number of cells to corrupt in each column
    corrupted_cells = round(df.shape[0] * corruption_rate)

    for j in continuous_columns:
        # percentile rank of each value (ties share a rank), always > 0
        rank = df[j].rank(method="average", pct=True).to_numpy()

        # weight grows with the rank, strength controls how strongly
        weights = rank ** self_masking_strength
        weights = weights / weights.sum()

        # weighted sample without replacement
        rows = rng.choice(df.index.to_numpy(), size=corrupted_cells, replace=False, p=weights)

        corrupted_df.loc[rows, j] = None
        mask_df.loc[rows, j] = True

    return corrupted_df, mask_df

# label correlated corruptor (mar given the label: missingness depends on the observed target)
def label_correlated_missing(df, corruption_rate, seed):

    corrupted_df = df.copy() # creaet a copy of df to corrupt

    eligible_cells = df.shape[0] * len(continuous_columns)

    # create a df to fill with Fase (non empty cells)

    mask_df = pd.DataFrame(False, index=df.index, columns=avaliable_columns+[target_column])

    high_usage_sample = []
    low_usage_sample = []

    # get all the possible pearis row + column
    for i in df.index:
        for j in continuous_columns:
            if corrupted_df.at[i, target_column] == 1:
                high_usage_sample.append((i, j))
            else:
                low_usage_sample.append((i, j))

    rng = random.Random(seed)

    high_usage_proportion = corrupted_df[target_column].sum() / df.shape[0]
    low_usage_proportion = (df.shape[0] - corrupted_df[target_column].sum()) / df.shape[0]

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
    eligible_columns = typo_columns
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

def decimal_corruptor(df, corruption_rate, seed):
    eligible_columns = decimal_columns

    corrupted_df = df.copy() # creaet a copy of df to corrupt
    # create a df to fill with Fase (non empty cells)
    mask_df = pd.DataFrame(False, index=df.index, columns=avaliable_columns+[target_column])
    sample_list = [] # creating all possible pairs based on rows and columns

    # get all the possible pearis row + column
    for i in df.index:
        for j in eligible_columns:
            if corrupted_df.at[i, j] != 0:
                sample_list.append((i, j))

    corrupted_cells = round(len(sample_list) * corruption_rate) #number of cells that has to be corrupted

    rng = random.Random(seed)

    # simple random sample ne = corrupted_cells
    random_sample_list = rng.sample(sample_list, corrupted_cells)

    # corrupt cells
    for i in random_sample_list:
        probability = rng.randint(0, 1)
        if probability == 1:
            corrupted_df.at[i[0], i[1]] = corrupted_df.at[i[0], i[1]] * 10
        else:
            corrupted_df.at[i[0], i[1]] = corrupted_df.at[i[0], i[1]] / 10
        mask_df.at[i[0], i[1]] = True

    return corrupted_df, mask_df

def duplicates_corruptor(df, corruption_rate, seed):

    duplicates_needed = round(df.shape[0] * corruption_rate)
    sample_list = [] # creating all possible pairs based on rows and columns

    corrupted_df = df.copy()

    for i in corrupted_df.index:
        sample_list.append(i)

    rng = random.Random(seed)

    random_sample_list = rng.sample(sample_list, duplicates_needed)

    mask_columns = ["new_row_id", "source_row_id", "perturbed_cols"]
    mask_df = pd.DataFrame(index=range(duplicates_needed), columns=mask_columns)

    row_count = 0
    for i in random_sample_list:
        prob = rng.randint(0, 1)

        if prob == 0:
            column_name = [rng.choice(continuous_columns)]
        else:
            # choose 2 different continuous columns
            column_name = rng.sample(continuous_columns, 2)

        new_row_id = df.index.max() + 1 + row_count

        corrupted_df.loc[new_row_id] = corrupted_df.loc[i]
        corrupted_df.loc[new_row_id, column_name] = corrupted_df.loc[new_row_id, column_name] + 0.00001

        mask_df.loc[row_count, "new_row_id"] = new_row_id
        mask_df.loc[row_count, "source_row_id"] = i
        mask_df.loc[row_count, "perturbed_cols"] = column_name

        row_count += 1

    return corrupted_df, mask_df








