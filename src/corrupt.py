# Here you can find the functions that corrupt the data set

import pandas as pd # import pandas
import random # import random for corrupting 
from pathlib import Path
import yaml

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

# load configuration file
with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_drug = config["target_drug"]

avaliable_columns = config["columns"]["continuous"] + config["columns"]["categorical"]
continuous_columns = config["columns"]["continuous"]

# avaliable_columns = [
#     'Age',
#     'Gender', 
#     'Education', 
#     'Country', 
#     'Ethnicity', 
#     'Nscore',
#     'Escore',
#     'Oscore', 
#     'AScore', 
#     'Cscore', 
#     'Impulsive', 
#     'SS',
# ]

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




