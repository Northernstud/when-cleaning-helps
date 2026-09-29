# import pandas
import pandas as pd 
from pathlib import Path
import yaml

# get the paths
project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"
drug_consumption = project_dir / "data/raw/Drug_Consumption.csv"
processed_dir = project_dir / "data/processed/"

# load configuration file
with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_drug = config["target_drug"]
columns = config["columns"]

features = columns["continuous"] + columns["categorical"]

# read the dataset
def load_model_data(file_path: Path): 

    df = pd.read_csv(file_path)

    # get the features and target drug
    df = df[features + [target_drug]].copy()

    # check if the value is CL3 CL4 CL5 CL6 if so True otherwise False then we turn the column type to int so it is 1 and 0
    positive_classes = ["CL3", "CL4", "CL5", "CL6"]

    df[target_drug] = df[target_drug].isin(positive_classes).astype(int)
    

    return df

# save the dataset in data/processed/
load_model_data(drug_consumption).to_csv(processed_dir / f"Drug_Consumption_{target_drug}.csv", index=False)
