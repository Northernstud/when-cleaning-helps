# Here you can find fucntions that clean the dataset
# every cleaner learns from the train df only (means, imputers, canonical labels)
# and applies the same thing to the holdout if one is given -> returns (cleaned_df, cleaned_holdout)
from pathlib import Path

import pandas as pd
import yaml
from rapidfuzz import fuzz, process
from sklearn.impute import KNNImputer

project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

# fille the empty values with mean of that column
def mean_fill(df, cols, params, holdout_df=None):

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

    clean_holdout = None
    if holdout_df is not None:
        clean_holdout = holdout_df.copy()
        # fill the holdout with the train means
        for i in cols:
            clean_holdout[i] = clean_holdout[i].fillna(means[i])

    return clean_df, clean_holdout

# mean fill + one 0/1 column per continuous column that says "this value was missing"
def mean_fill_indicator(df, cols, params, holdout_df=None):

    continuous_columns = cols["continuous"]

    # build the flags before filling, otherwise there is nothing left to flag
    flags = df[continuous_columns].isna().astype(int).add_suffix("_missing")
    if holdout_df is not None:
        holdout_flags = holdout_df[continuous_columns].isna().astype(int).add_suffix("_missing")

    clean_df, clean_holdout = mean_fill(df, cols, params, holdout_df)

    # flags are added even when a column has no missing values so train and holdout have the same columns
    clean_df = pd.concat([clean_df, flags], axis=1)
    if holdout_df is not None:
        clean_holdout = pd.concat([clean_holdout, holdout_flags], axis=1)

    return clean_df, clean_holdout

def do_nothing(df, cols, params, holdout_df=None):
    clean_holdout = None if holdout_df is None else holdout_df.copy()
    return df.copy(), clean_holdout

# fit a sklearn imputer on the train continuous columns and apply it to both dfs
def fit_imputer(imputer, df, cols, holdout_df):

    cleaned_df = df.copy()

    continuous_columns = cols["continuous"]

    cleaned_df[continuous_columns] = imputer.fit_transform(
        df[continuous_columns]
    )

    cleaned_holdout = None
    if holdout_df is not None:
        cleaned_holdout = holdout_df.copy()
        cleaned_holdout[continuous_columns] = imputer.transform(
            holdout_df[continuous_columns]
        )

    return cleaned_df, cleaned_holdout

def knn_impute(df, cols, params, holdout_df=None):

    k = params["n_neighbors"]

    imputer = KNNImputer(n_neighbors=k)

    return fit_imputer(imputer, df, cols, holdout_df)

# rows can be dropped from the train set, but a model can not refuse to predict a holdout row
# so the holdout is left as is
def drop_rows(df, cols, params, holdout_df=None):
    clean_holdout = None if holdout_df is None else holdout_df.copy()
    return df.dropna().copy(), clean_holdout

def fuzzy_match(df, cols, params, holdout_df=None):
    cleaned_df = df.copy()
    categorical_columns = cols["categorical"]
    threshold = params.get("threshold", 85)
    scorer = params.get("scorer", fuzz.ratio)
    mappings = {}

    cleaned_holdout = None if holdout_df is None else holdout_df.copy()

    for col in categorical_columns:
        norm = cleaned_df[col].astype("string").str.strip().str.lower()

        # normalized form -> the original-cased text to write back
        # use the most frequent spelling, otherwise a case-only typo ("JUly") seen first would rename the whole category
        display = {}
        spellings = pd.DataFrame({"norm": norm, "orig": cleaned_df[col]}).dropna().value_counts()  # most frequent first
        for n, orig in spellings.index:
            if n not in display:
                display[n] = orig
        counts = norm.value_counts()  # most frequent first

        canonical = []
        col_map = {}

        for value in counts.index:
            result = process.extractOne(value, canonical,
                                        scorer=scorer, score_cutoff=threshold)
            if result is not None:
                match, score, _ = result
                col_map[value] = display[match]               # rare value -> trusted spelling
            else:
                canonical.append(value)        # new real category
                col_map[value] = display[value]

        cleaned_df[col] = norm.map(col_map)
        mappings[col] = col_map

        if holdout_df is not None:
            # map the holdout with the categories learned on train
            holdout_norm = cleaned_holdout[col].astype("string").str.strip().str.lower()
            holdout_map = {}
            for value in holdout_norm.dropna().unique():
                if value in col_map:
                    holdout_map[value] = col_map[value]
                    continue
                result = process.extractOne(value, canonical,
                                            scorer=scorer, score_cutoff=threshold)
                if result is not None:
                    match, score, _ = result
                    holdout_map[value] = display[match]
                # no match -> unseen value, not in the map so it is left as it was

            # write back only the mapped values, keep the rest as they were
            mapped = holdout_norm.map(holdout_map)
            cleaned_holdout[col] = mapped.where(mapped.notna(), cleaned_holdout[col])

    return cleaned_df, cleaned_holdout
