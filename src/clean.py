# Here you can find fucntions that clean the dataset
import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
import yaml 

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

# fille the empty values with mean of that column
def mean_fill(df, cols, params):

    # get the continuos data from cols 
    cols = cols["continuous"]

    # hash map with all the means for each col
    means = {}

    clean_df = df.copy() # make a coppy of corrupted df to clean

    # calculate mean for each value
    for i in cols: 
        means[i] = clean_df[i].mean()

    # go through each cell to check if it is NaN then replce with mean
    for i in cols: 
        clean_df[i] = clean_df[i].fillna(means[i])

    return clean_df

def do_nothing(df, cols, params):
    return df.copy()

def knn_impute(df, cols, params):

    cleaned_df = df.copy()

    continuous_columns = cols["continuous"]

    k = params["n_neighbors"]

    imputer = KNNImputer(n_neighbors=k)

    cleaned_df[continuous_columns] = imputer.fit_transform(
        df[continuous_columns]
    )

    return cleaned_df
    

    