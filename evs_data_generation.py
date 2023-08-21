import tomllib
import pandas as pd
import numpy as np

CVS_PATH = "data/EVs.csv"
TOML_PATH = "config/params.toml"

with open(TOML_PATH, "rb") as file: s = tomllib.load(file)

evs_df = pd.read_csv(CVS_PATH) 
for coeff, lim in s["coeff"].items():
    evs_df[coeff] = np.random.uniform(lim["min"], lim["max"], size=evs_df.shape[0]).round(4)

evs_df.to_csv(CVS_PATH, index=False)