# import pandas
import pandas as pd 
from pathlib import Path
import yaml

# get the paths
project_dir = Path(__file__).resolve().parent.parent
config_path = project_dir / "config.yaml"
csv = project_dir / "data/raw/hotel_bookings.csv"
processed_dir = project_dir / "data/processed/"

# load configuration file
with config_path.open() as config_file:
    config = yaml.safe_load(config_file)

target_column = config["target_column"]
columns = config["columns"]

features = columns["continuous"] + columns["categorical"]

# read the dataset
def load_model_data(file_path: Path): 

    df = pd.read_csv(file_path)

    # get the features and target drug
    df = df[features + [target_column]].copy()
    df= df.drop_duplicates()

    df = df.dropna(subset=["children"])          # only ~4 rows
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(-1).astype(int).astype(str).replace("-1", "None")
    df = df[(df["adults"] + df["children"]) > 0]
    df = df[(df["adr"] >= 0) & (df["adr"] < 1000)]
    df["children"] = df["children"].astype(int)


    df.to_csv(processed_dir / f"hotel_bookings_clean.csv", index=False)
    return df

# save the dataset in data/processed/
load_model_data(csv).to_csv(processed_dir / f"csv_{target_column}.csv", index=False)
