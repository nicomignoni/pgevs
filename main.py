import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from model import System, EvolutionaryDynamic, UniformDynamic

np.random.seed(2023)

NUM_RUNS = 1
CONFIG_PATH = "config/params.json"

# Utilities
def get_evs_sample(n, evs_df, css_df, settings):
    x_df = pd.concat([
        css_df.sample(n, replace=True, ignore_index=True), 
        evs_df.sample(n, replace=True, ignore_index=True)], 
        axis=1
    )
    x_df["max_power"] = np.minimum(x_df["max_input"].values, x_df["max_output"].values)
    x_df["soc"] = settings["init_soc_frac"] * x_df["max_soc"] * np.random.rand(n)
    x_df["alpha"] = np.random.uniform(
        settings["coeff"]["alpha"]["min"], 
        settings["coeff"]["alpha"]["max"], 
        size=n
    )
    return x_df


# Load data
with open(CONFIG_PATH, "r") as file: settings = json.load(file)
css_df = pd.read_csv("data/CSs.csv")
evs_df = pd.read_csv("data/EVs.csv")

"""Simulations"""
timestamp = datetime.now().strftime('%d-%m-%YT%H-%M-%S')
os.mkdir(f"results/{timestamp}/")
for i in range(NUM_RUNS):
    print(f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')} - Running {i+1}-th simulation...", end=" ")

    # Sample EVs and CSs data
    settings["init_frac_empty_css"] = 1 # np.random.rand()
    num_init_empty_css = int(settings["init_frac_empty_css"] * settings["num_cs"])
    empty_css_indices = np.random.randint(0, settings["num_cs"], num_init_empty_css)
    x_df = get_evs_sample(settings["num_cs"], evs_df, css_df, settings)
    x_df.iloc[empty_css_indices] = 0

    # Add retailer row
    x_df.loc[settings["num_cs"]] = np.zeros(x_df.shape[1])
    x_df.loc[settings["num_cs"], ("max_output", "max_input", "max_power")] = settings["avail"]
    x_df.loc[settings["num_cs"], "is_active"] = 1

    # Initial values
    init_x = np.array(x_df.to_records(index=False))
    init_x["power"] = System.uniform_power_allocation(init_x, settings)

    # Instantiate dynamics
    edyn = EvolutionaryDynamic(init_x, settings)
    udyn = UniformDynamic(init_x, settings)

    # System loop
    num_steps = int(settings["t_bound"] / settings["step_size"])
    for _ in range(num_steps):
        # Departures
        has_departed = np.random.uniform(size=settings["num_cs"]) <= settings["dep_prob"]
        for field in ["soc", "max_soc", "max_power"] + list(settings["coeff"].keys()): 
            edyn.x[field][edyn.k,:-1] *= ~has_departed
            udyn.x[field][udyn.k,:-1] *= ~has_departed

        # Arrivals (same for the evolutionary and uniform dynamic)
        has_arrived = np.logical_and(
            edyn.x["max_soc"][edyn.k,:-1] == 0, 
            np.random.uniform(size=settings["num_cs"]) <= settings["arr_prob"]
        )
        arrived_evs_indices = np.flatnonzero(has_arrived)
        if (num_arrived_evs := arrived_evs_indices.size) >= 1:
            arrived_evs_data = get_evs_sample(num_arrived_evs, evs_df, css_df, settings)
            edyn.x[:][edyn.k,arrived_evs_indices] = np.array(arrived_evs_data.to_records(index=False))
            udyn.x[:][edyn.k,arrived_evs_indices] = np.array(arrived_evs_data.to_records(index=False))

        # Jump and flow dynamics
        edyn.make_flow_step()
        udyn.make_flow_step()

    # Save run
    np.savez(f"results/{timestamp}/run_{i}.npz", edyn_x=edyn.x, udyn_x=udyn.x); print("Done")

# Save a copy of the configuration file with the results
with open(f"results/{timestamp}/params.json", "w") as file: json.dump(settings, file)