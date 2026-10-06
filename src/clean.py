# Here you can find fucntions that clean the dataset
import pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import KNNImputer, IterativeImputer
from rapidfuzz import process, fuzz
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

def drop_null_rows(df, cols, params):

    cleaned_df = df.copy()

    cleaned_df = cleaned_df.dropna()

    return cleaned_df

def fuzzy_match(df, cols, params):
    cleaned_df = df.copy()
    categorical_columns = cols["categorical"]
    threshold = params.get("threshold", 85)        
    scorer = params.get("scorer", fuzz.ratio)
    mappings = {}

    for col in categorical_columns:
        cleaned_df[col] = cleaned_df[col].astype("string").str.strip().str.lower()
        counts = cleaned_df[col].value_counts()  # most frequent first

        canonical = []
        col_map = {}

        for value in counts.index:
            result = process.extractOne(value, canonical,
                                        scorer=scorer, score_cutoff=threshold)
            if result is not None:
                match, score, _ = result
                col_map[value] = match               # rare value -> trusted spelling
            else:
                canonical.append(value)              # new real category

        cleaned_df[col] = cleaned_df[col].replace(col_map)
        mappings[col] = col_map

    return cleaned_df, mappings
