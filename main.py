import os
import json
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from model import System, EvolutionaryDynamic, UniformDynamic
from plotter import Plotter

np.random.seed(2023)

NUM_RUNS = 10
RUN_SIMULATIONS = False
CONFIG_PATH = "config/params.json"
SIMULATIONS_PATH = "results/22-08-2023T12-26-19/run.npz"

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

# Load matplotlib settings
with open("config/matplotlib.json", "r") as file: plts = json.load(file)
plts["rcParams"]["text.latex.preamble"] = "".join(plts["latex_preamble"])
plt.rcParams.update(**plts["rcParams"])

# Load data
with open(CONFIG_PATH, "r") as file: settings = json.load(file)
settings["num_runs"] = NUM_RUNS
css_df = pd.read_csv("data/CSs.csv")
evs_df = pd.read_csv("data/EVs.csv")

"""Simulations"""
if RUN_SIMULATIONS:
    num_steps = int(settings["t_bound"] / settings["step_size"])

    timestamp = datetime.now().strftime('%d-%m-%YT%H-%M-%S')
    os.mkdir(f"results/{timestamp}/")

    # KPIs to retain
    kpis = ["fairness", "collective_charging", "availability_use"]
    edyn_kpis = np.zeros([NUM_RUNS, num_steps+1], dtype=list(product(kpis, ["<f8"])))
    udyn_kpis = np.zeros([NUM_RUNS, num_steps+1], dtype=list(product(kpis, ["<f8"])))

    for i in range(NUM_RUNS):
        print(f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')} - Running {i+1}-th simulation...", end=" ")

        # Sample EVs and CSs data
        settings["init_frac_empty_css"] = np.random.rand()
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
        for _ in range(num_steps):
            # Departures (same for the evolutionary and uniform dynamic)
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

        # Collect KPIs
        for kpi in kpis:
            edyn_kpis[kpi][i,:] = getattr(edyn, kpi)
            udyn_kpis[kpi][i,:] = getattr(udyn, kpi)

        # Save data
        np.savez(f"results/{timestamp}/run.npz", edyn_kpis=edyn_kpis, udyn_kpis=udyn_kpis); print("Done")

# Analize KPIs
edyn, udyn = EvolutionaryDynamic(0, settings), UniformDynamic(0, settings)
edyn.kpis, udyn.kpis = np.load(SIMULATIONS_PATH).values()

def plot_kpi(kpi, systems, plot_label, y_label, **kwargs):
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    for system in systems:
        avg_kpi = system.kpis[kpi].mean(0)
        std_kpi = system.kpis[kpi].std(0)
        ax.fill_between(system.t, avg_kpi - std_kpi, avg_kpi + std_kpi, color=system.color, alpha=0.2)
        ax.plot(system.t, avg_kpi, label=plot_label % {"sym": system.var_symbol}, color=system.color)
    ax.grid(); ax.set_xlabel("Time [h]"); ax.set_ylabel(y_label)
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=4)
    plt.tight_layout()
    plt.savefig(kwargs["save_path"], bbox_inches="tight")
    plt.show(block=False)

def boxplot_kpi(kpi, systems, x_ticks, y_label, **kwargs):
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.boxplot([system.kpis[kpi].sum(1) for system in systems])
    plt.xticks(np.arange(1,len(systems)+1), x_ticks)
    ax.grid(); ax.set_ylabel(y_label); ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig(kwargs["save_path"], bbox_inches="tight")
    plt.show(block=False)

# Plot KPIs
plot_kpi(
    "fairness", (edyn, udyn), 
    y_label="€",
    plot_label=r"$\text{std}(\mathbf{g}%(sym)s(t))$",
    figsize=(plts["fig_width"], 0.55*plts["fig_width"]), 
    save_path="fig/avg_fairness.pdf"
)
plot_kpi(
    "collective_charging", (edyn, udyn), 
    y_label="\%",
    plot_label=r"$\phi%(sym)s(t)$",
    figsize=(plts["fig_width"], 0.55*plts["fig_width"]), 
    save_path="fig/avg_collective_charging.pdf"
)
plot_kpi(
    "availability_use", (edyn, udyn), 
    y_label="\%",
    plot_label=r"$\sum_{i \in \mathcal{C}} p_i%(sym)s(t)$",
    figsize=(plts["fig_width"], 0.55*plts["fig_width"]), 
    save_path="fig/avg_availability_use.pdf"
)

# Boxplot KPIs
boxplot_kpi(
    "fairness", (edyn, udyn),
    x_ticks=["Evolutionary \n dynamic", "Uniform \n dynamic ($*=\\dagger$)"],
    y_label=r"$\sum_{t} \text{std}(g_i^*(t))$ [€]",
    figsize=(0.5*plts["fig_width"], 0.5*plts["fig_width"]),
    save_path="fig/box_fairness.pdf"
)
boxplot_kpi(
    "collective_charging", (edyn, udyn),
    x_ticks=["Evolutionary \n dynamic", "Uniform \n dynamic ($*=\\dagger$)"],
    y_label=r"$\sum_{t} \phi_i^*(t)$ [\%]",
    figsize=(0.5*plts["fig_width"], 0.5*plts["fig_width"]),
    save_path="fig/box_collective_charging.pdf"
)