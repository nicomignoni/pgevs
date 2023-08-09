import tomllib
import pandas as pd
import numpy as np

csv_path = "data/EVs.csv"
toml_path = "python/params.toml"

with open(toml_path, "rb") as file: s = tomllib.load(file)

evs_df = pd.read_csv(csv_path) 
for coeff, lim in s["coeff"].items():
    evs_df[coeff] = np.random.uniform(lim["min"], lim["max"], size=evs_df.shape[0]).round(4)

evs_df.to_csv(csv_path, index=False)