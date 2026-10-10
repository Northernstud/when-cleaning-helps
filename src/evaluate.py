# Here you can find functions that evaluate cleaning methods
import pandas as pd 
import numpy as np 
import yaml
from pathlib import Path
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"

with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

columns = config["columns"]
target_drug = config["target_drug"]
corruption_families = config["corruption_families"]

NOT_APPLICABLE = {
        "rmse": None,
        "mae": None,
        "bias": None,
        "variance_ratio": None,
        "wasserstein": None,
        "recovery_applicable": False
    }  

CAT_NOT_APPLICABLE = {
        "cat_match_rate": None,
        "cat_unresolved_rate": None,
        "cat_wrong_label_rate": None,
        "cat_recovery_applicable": None,
    }

def evaluate(original_df, dirty_df, cleaned_df, mask, corruption_type, cols, target_drug):
    out = empty_metric_template()
    family = corruption_families[corruption_type]
    if family == "continuous":
        out.update(evaluate_continuous_recovery(original_df, cleaned_df, mask, cols))
    elif family == "categorical":
        out.update(evaluate_categorical_recovery(original_df, cleaned_df, mask, dirty_df, cols))
    # elif family == "row":
    #     out.update(evaluate_detection_quality(mask, cleaned_df))
    out.update(evaluate_downstream_performance(cleaned_df, target_drug))
    return out

def empty_metric_template():
    return {
        "rmse": None,
        "mae": None,
        "bias": None,
        "variance_ratio": None,
        "wasserstein": None,
        "recovery_applicable": None,
        "cat_match_rate": None,
        "cat_nan_rate": None,
        "cat_untouched_rate": None,
        "cat_wrong_label_rate": None,
        "cat_recovery_applicable": None,
        "detect_precision": None,
        "detect_recall": None,
        "detect_f1": None,
        "detect_applicable": None,
        "roc_auc": None,
        "roc_auc_std": None,
        "accuracy": None,
        "f1": None,
        "n_rows_used": None,
    }

def evaluate_continuous_recovery(original_df, cleaned_df, mask_df, cols):
    
    # create a copy of dfs 
    original_df = original_df.copy()
    cleaned_df = cleaned_df.copy()
    mask_df = mask_df.copy()

    cols = cols["continuous"]

    # data from original and from cleaned df

    recovery_applicable = True

    common = mask_df.index.intersection(cleaned_df.index)
    m = mask_df.loc[common, cols].to_numpy()

    if m.sum() == 0:
        return NOT_APPLICABLE

    true_values = original_df.loc[common, cols].to_numpy()[m]
    imputed_values = cleaned_df.loc[common, cols].to_numpy()[m]

    if np.isnan(imputed_values).any():
        return NOT_APPLICABLE
    
    residuals = true_values - imputed_values

    # calculate rmse
    rmse = np.sqrt(np.mean(residuals ** 2))

    # claculate mae
    mae = np.mean(np.abs(residuals))

    # calculate bias
    bias = np.mean(residuals)

    # calculate variance ratio 
    if len(imputed_values) < 2 or np.var(true_values) == 0:
        variance_ratio = None
    else:
        variance_ratio = np.var(imputed_values) / np.var(true_values)

    # Wasserstein distance
    wasserstein = wasserstein_distance(imputed_values, true_values)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "variance_ratio": variance_ratio,
        "wasserstein": wasserstein,
        "recovery_applicable": recovery_applicable
    }

def evaluate_categorical_recovery(original_df, cleaned_df, mask_df, dirty_df, cols):
     # create a copy of dfs 
    original_df = original_df.copy()
    cleaned_df = cleaned_df.copy()
    mask_df = mask_df.copy()

    cols = ["Education", "Country"]

    common = mask_df.index.intersection(cleaned_df.index)
    m = mask_df.loc[common, cols].to_numpy()


    if m.sum() == 0:
        return CAT_NOT_APPLICABLE

    
    true_values = original_df.loc[common, cols].to_numpy()[m]
    imputed_values = cleaned_df.loc[common, cols].to_numpy()[m]
    nans = pd.isna(imputed_values).sum()

    total_corrupted = m.sum()

    matches = (true_values == imputed_values).sum()
    dirty_values = dirty_df.loc[common, cols].to_numpy()[m]
    non_matches = total_corrupted - matches
    non_nan = ~pd.isna(imputed_values)
    untouched = ((imputed_values == dirty_values) & non_nan & (true_values != imputed_values)).sum()

    wrong_label = total_corrupted - matches - nans - untouched

    cat_recovery_applicable = True



    # match rate
    cat_match_rate      = float(matches / total_corrupted)
    cat_nan_rate        = float(nans / total_corrupted)
    cat_untouched_rate  = float(untouched / total_corrupted)
    cat_wrong_label_rate = float(wrong_label / total_corrupted)

    return {
        "cat_match_rate": cat_match_rate,
        "cat_untouched_rate": cat_untouched_rate,
        "cat_wrong_label_rate": cat_wrong_label_rate,
        'cat_nan_rate': cat_nan_rate,
        "cat_recovery_applicable": cat_recovery_applicable,
    }


def evaluate_downstream_performance(cleaned_df, target_column): 

    # initiate the explanatory and response dfs
    X = cleaned_df.drop(target_column, axis=1)
    y = cleaned_df[target_column]

    # split the train and test dfs 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define the separate categorical pipeline
    categorical_transformer = Pipeline(
        steps=[
            # First, fill missing values with constant string
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            # Then, apply one-hot encoding to the clean string values
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # transform the categorical data into numerical representation
    transformer = ColumnTransformer(transformers=[
        ('tnf1', categorical_transformer, columns["categorical"])
    ], remainder='passthrough')

    model = Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("classifier", HistGradientBoostingClassifier(
                random_state=42
            ))
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # calculate the ROC AUC score
    auc_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    clf = model.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    # Find the column corresponding to class 1
    positive_class_index = list(clf.classes_).index(1)

    # Extract probabilities for class 1
    y_proba_positive = y_proba[:, positive_class_index]

    # calculate roc auc
    auc_score = roc_auc_score(y_test, y_proba_positive)

    # calculate the ROC AUC stdv 
    roc_auc_std = np.std(auc_scores, ddof=1)

    # calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # calculate f1 score
    f1 = f1_score(y_test, y_pred)

    # row used
    n_rows_used = len(y_test)

    return {
        "roc_auc": auc_score,
        "roc_auc_std": roc_auc_std,
        "accuracy": accuracy,
        "f1": f1,
        "n_rows_used": n_rows_used,
    }


