import os
import json
from itertools import product

import numpy as np

import matplotlib.pyplot as plt
from model import EvolutionaryDynamic, UniformDynamic
from plotter import Plotter

SIMULATIONS_PATH = "results/22-08-2023T01-45-50" # <- put the results you want to plot here


# Figure and system settings
with open(f"{SIMULATIONS_PATH}/params.json", "r") as file: settings = json.load(file)

# Plotter settings
with open("config/matplotlib.json", "r") as file: plts = json.load(file)



num_steps = int(settings["t_bound"] / settings["step_size"])
t = settings["step_size"] * np.arange(num_steps+1)

edyn = EvolutionaryDynamic(0, settings)
udyn = UniformDynamic(0, settings)

edyn.kpis, udyn.kpis = np.load(f"{SIMULATIONS_PATH}/run.npz").values()
    
# Fairness 
fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.55*plts["fig_width"]))
for system in (edyn, udyn):
    ax.plot(
        t, system.kpis["fairness"][9,:], 
        label=r"$\text{std}(\mathbf{g}" + f"{system.var_symbol}(t))$",
        color=system.color, 
        linewidth=0.9
    )
# plot_css_traffic(ax, t, has_any_ev_departed, "x")
# plot_css_traffic(ax, t, has_any_ev_arrived, "o")
ax.set_xlabel("Time [h]")
ax.set_ylabel("€")
ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=4)
ax.grid()
plt.tight_layout()
plt.savefig("fig/precedence.pdf", bbox_inches="tight")
plt.show(block=False) 

# # Load results and use the "worst" one as example

# edyn = EvolutionaryDynamic(0, settings)
# udyn = UniformDynamic(0, settings)

# edyn.x, udyn.x = np.load(f"{SIMULATIONS_PATH}/run_0.npz").values()

# # # (Debugging)
# # plot_trajectory(t, edyn.x)
# # plot_trajectory(t, udyn.x)

# # Get departures (same for evolutionary and uniform)
# has_any_ev_departed = np.logical_and(edyn.x["max_soc"][:-1,:] > 0, edyn.x["max_soc"][1:,:] == 0).any(1)
# has_any_ev_arrived = np.logical_and(edyn.x["max_soc"][:-1,:] == 0, edyn.x["max_soc"][1:,:] > 0).any(1)

# # Collective charging and availability utilization
# fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.55*plts["fig_width"])) 
# for system in (edyn, udyn):
#     ax.plot( 
#         t, 100 * system.collective_charging(), 
#         label=r"$\phi" + f"{system.var_symbol}(t)$", ls="--",
#         color=system.color, 
#         linewidth=0.9
#     )
#     ax.plot(
#         t, 100 * system.availability_utilization(), 
#         label=r"$\sum_{i \in \mathcal{C}} p_i" + f"{system.var_symbol}(t)$", 
#         color=system.color, 
#         linewidth=0.9
#     )
# plot_css_traffic(ax, t, has_any_ev_departed, "x")
# plot_css_traffic(ax, t, has_any_ev_arrived, "o")
# ax.set_xlabel("Time [h]")
# ax.set_ylabel(r"[\%]")
# ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4) 
# ax.grid()
# plt.tight_layout()
# plt.savefig("fig/charging_and_power.pdf", bbox_inches="tight")
# plt.show(block=False) 

# # Precedence function
# fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.55*plts["fig_width"]))
# for system in (edyn, udyn):
#     ax.plot(
#         t, system.precedence(slice(None)).std(axis=1), 
#         label=r"$\text{std}(\mathbf{g}" + f"{system.var_symbol}(t))$",
#         color=system.color, 
#         linewidth=0.9
#     )
# plot_css_traffic(ax, t, has_any_ev_departed, "x")
# plot_css_traffic(ax, t, has_any_ev_arrived, "o")
# ax.set_xlabel("Time [h]")
# ax.set_ylabel("€")
# ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=4)
# ax.grid()
# plt.tight_layout()
# plt.savefig("fig/precedence.pdf", bbox_inches="tight")
# plt.show(block=False) 