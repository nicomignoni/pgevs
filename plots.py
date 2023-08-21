import os
import json

import numpy as np

import matplotlib.pyplot as plt
from model import EvolutionaryDynamic, UniformDynamic

SIMULATIONS_PATH = "results/21-08-2023T02-12-34" # <- put the results you want to plot here

# Utilities
def plot_trajectory(t, x):
    # Plot the power and soc trajectories for EVs only
    fig, axs = plt.subplots(2, 1, figsize=(plts["fig_width"], 1.5*plts["fig_width"]), sharex=True)
    for i,(field, y_label) in enumerate([("power", "$p_i(t)$ [kW]"), ("soc", "$b_i(t)$ [kWh]")]):
        axs[i].plot(t, x[field][:,:-1], color="k", linewidth=0.3)
        axs[i].set_ylabel(y_label)
        axs[i].grid(alpha=0.2)
    axs[-1].set_xlabel("Time (h)")
    plt.show(block=False)

def plot_css_traffic(ax, t, arrival_or_departures, marker):
    alpha = 0.5
    time_indices = np.flatnonzero(arrival_or_departures)
    ymax = np.tile(ax.get_ylim()[1], time_indices.size)
    ax.vlines(t[time_indices], 0, ymax, lw=0.5, ls='--', color='k', alpha=alpha)
    ax.scatter(t[time_indices], ymax, marker=marker, c="k", s=10, alpha=alpha)


# Figure and system settings
with open(f"{SIMULATIONS_PATH}/params.json", "r") as file: settings = json.load(file)
with open("config/matplotlib.json", "r") as file: plts = json.load(file)
plts["rcParams"]["text.latex.preamble"] = "".join(plts["latex_preamble"])
plt.rcParams.update(**plts["rcParams"])

# Load results and use the "worst" one as example
num_steps = int(settings["t_bound"] / settings["step_size"])
t = settings["step_size"] * np.arange(num_steps+1)

edyn = EvolutionaryDynamic(0, settings)
udyn = UniformDynamic(0, settings)

edyn.x, udyn.x = np.load(f"{SIMULATIONS_PATH}/run_0.npz").values()

# (Debugging)
plot_trajectory(t, edyn.x)
plot_trajectory(t, udyn.x)

# Get departures (same for evolutionary and uniform)
has_any_ev_departed = np.logical_and(edyn.x["max_soc"][:-1,:] > 0, edyn.x["max_soc"][1:,:] == 0).any(1)
has_any_ev_arrived = np.logical_and(edyn.x["max_soc"][:-1,:] == 0, edyn.x["max_soc"][1:,:] > 0).any(1)

# Collective charging and availability utilization
fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.55*plts["fig_width"])) 
for system in (edyn, udyn):
    ax.plot( 
        t, 100 * system.collective_charging(), 
        label=r"$\phi" + f"{system.var_symbol}(t)$", ls="--",
        color=system.color, 
        linewidth=0.9
    )
    ax.plot(
        t, 100 * system.availability_utilization(), 
        label=r"$\sum_{i \in \mathcal{C}} p" + f"{system.var_symbol}(t)$", 
        color=system.color, 
        linewidth=0.9
    )
plot_css_traffic(ax, t, has_any_ev_departed, "x")
plot_css_traffic(ax, t, has_any_ev_arrived, "o")
ax.set_xlabel("Time [h]")
ax.set_ylabel(r"[\%]")
ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4) 
ax.grid()
plt.tight_layout()
plt.savefig("fig/charging_and_power.pdf", bbox_inches="tight")
plt.show(block=False) 

# Precedence function
fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.55*plts["fig_width"]))
for system in (edyn, udyn):
    ax.plot(
        t, system.precedence(slice(None)).std(axis=1), 
        label=r"$\text{std}(\mathbf{z}" + f"{system.var_symbol}(t))$",
        color=system.color, 
        linewidth=0.9
    )
plot_css_traffic(ax, t, has_any_ev_departed, "x")
plot_css_traffic(ax, t, has_any_ev_arrived, "o")
ax.set_xlabel("Time [h]")
ax.set_ylabel("€")
ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=4)
ax.grid()
plt.tight_layout()
plt.savefig("fig/precedence.pdf", bbox_inches="tight")
plt.show(block=False) 