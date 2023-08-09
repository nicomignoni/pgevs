#%% Packages
import tomllib
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

np.random.seed(2023)

#%% Utilities
def efficiency(x): 
    return x["gamma"]*np.tanh(x["max_soc"]*x["sigma"] - x["epsilon"]*x["soc"]) + x["eta"] - x["gamma"]

def degradation(x): 
    return x["mu"] * x["power"]

def energy_cost(x): 
    return s["energy_price"] * x["power"][:ret_i].sum()

def emptyness(x): # 0 / 0 -> 0
    return np.divide(x["max_soc"] - x["soc"], x["max_soc"], out=np.zeros(num_agents), where=x["max_soc"] > 0)

# Events and handlers
def after_departures(x): 
    has_departed = np.random.uniform(size=num_agents) <= s["dep_prob"]
    for field in ("soc", "max_soc", "max_power", "is_active"): x[field] *= ~has_departed
    return x

def after_charge_optim(x):
    x["is_active"][:ret_i] = np.logical_and(x["max_soc"][:ret_i] > 0, x["soc"][:ret_i] <= x["max_soc"][:ret_i])
    realloc_power = np.sum(x["power"] * (1 - x["is_active"]))
    realloc_order = np.argsort(precedence(x))[::-1]
    max_power_increase = x["is_active"] * (x["max_power"] - x["power"])
    cum_power_increase = np.cumsum(max_power_increase[realloc_order])
    full_increase_evs = cum_power_increase <= realloc_power
    first_non_full_ev = np.argmin(full_increase_evs)
    x["power"][realloc_order] += full_increase_evs * max_power_increase[realloc_order]
    x["power"][realloc_order[first_non_full_ev]] += realloc_power - max_power_increase[realloc_order[:first_non_full_ev]].sum()
    x["power"] *= x["is_active"]
    return x
    
def after_charge_simple(x):
    x["is_active"][:ret_i] = np.logical_and(x["max_soc"][:ret_i] > 0, x["soc"][:ret_i] <= x["max_soc"][:ret_i])
    x["power"][ret_i] += np.sum(x["power"][:ret_i] * (1 - x["is_active"][:ret_i]))
    x["power"] *= x["is_active"]
    return x

def after_arrivals(x):
    has_arrived = np.logical_and(x["is_active"] == 0, np.random.uniform(size=num_agents) <= s["arr_prob"])
    num_arrived_evs = np.count_nonzero(has_arrived)
    if num_arrived_evs >= 1:
        new_evs = evs_df.sample(num_arrived_evs, replace=True, ignore_index=True).to_records(index=False)
        for field in new_evs.dtype.names:
            x[field][has_arrived] = new_evs[field]
        x["max_power"] = np.minimum(x["max_input"], x["max_output"])
    return x

# Fitness and revision
def precedence(x): 
    return x["alpha"] * emptyness(x) 

def rho(x):
    g = precedence(x) 
    return x["is_active"] * x["is_active"][:,None] * np.maximum(0, x["max_power"][:,None] - x["power"][:,None])*\
           np.maximum(0, g[:,None] - g)

# Dynamics
def dynamic_eg(x): 
    """Jump dynamic"""
    # x = after_departures(x)
    x = after_charge_optim(x)
    # x = after_arrivals(x)

    """Flow dynamic"""
    r = x["power"] * rho(x)
    x["power"] += (np.sum(r,1) - np.sum(r,0)) * s["step_size"] 
    x["soc"] += (efficiency(x) * x["power"]) * s["step_size"] 
    return x

def dynamic_tr(x):
    """Jump dynamic"""
    # x = after_departures(x)
    x = after_charge_simple(x)
    # x = after_arrivals(x)

    uniform_allocation = np.divide(s["avail"], 
        x["is_active"][:ret_i].sum(),
        where=x["is_active"][:ret_i]==1,
        out=np.zeros(s["num_cs"])
    ) 
    x["power"][ret_i] = 0
    x["power"][:ret_i] = np.minimum(uniform_allocation, x["max_power"][:ret_i])
    x["power"][ret_i] = s["avail"] - x["power"][:ret_i].sum()
    x["soc"] += (efficiency(x) * x["power"]) * s["step_size"] 
    return x


#%% Figure and system settings
with open("matplotlib.toml", "rb") as file: plts = tomllib.load(file)
with open("params.toml", "rb") as file: s = tomllib.load(file)
plts["rcParams"]["text.latex.preamble"] = "".join(plts["latex_preamble"])
plt.rcParams.update(**plts["rcParams"])

num_agents = s["num_cs"] + 1
num_steps = int(s["t_bound"] / s["step_size"])
ret_i = s["num_cs"]

# CSs data are invariant with respect to any dynamic
css_df = pd.read_csv("../data/CSs.csv", usecols=lambda x: x != "kind")
evs_df = pd.read_csv("../data/EVs.csv", usecols=lambda x: x != "name")
x_df = pd.concat([
    css_df.sample(s["num_cs"], replace=True, ignore_index=True), 
    evs_df.sample(s["num_cs"], replace=True, ignore_index=True)], 
    axis=1
)

# Add retailer row
x_df.loc[ret_i] = np.zeros(x_df.shape[1])
x_df.loc[ret_i, ("max_output", "max_input")] = s["avail"]
x_df.loc[ret_i, "is_active"] = 1

# Add max power 
x_df["max_power"] = np.minimum(x_df["max_input"].values, x_df["max_output"].values)

# Initial values
init_x = np.array(x_df.to_records(index=False))
init_power_alloc = np.divide(
    s["avail"], 
    init_x["is_active"][:ret_i].sum(),
    where=init_x["is_active"][:ret_i]==1,
    out=np.zeros(s["num_cs"])
) 
init_x["soc"] = s["init_soc_frac"] * init_x["max_soc"] * np.random.rand(num_agents)
init_x["power"][:ret_i] = np.minimum(init_power_alloc, init_x["max_power"][:ret_i])
init_x["power"][ret_i] = 0

# Evolutionary dynamic block
x_eg = np.zeros([num_steps, num_agents], dtype=init_x.dtype)
x_eg[:][0,:] = init_x

# Trivial dynamic block
x_tr = np.zeros([num_steps, num_agents], dtype=init_x.dtype)
x_tr[:][0,:] = init_x


#%% System loop
t = s["step_size"] * np.arange(num_steps)
for k in range(1,num_steps):
    x_eg[:][k,:] = dynamic_eg(x_eg[:][k-1,:])
    x_tr[:][k,:] = dynamic_tr(x_tr[:][k-1,:])

# Save results
# timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
np.save("../results/output_eg.npy", x_eg)
np.save("../results/output_tr.npy", x_tr)
np.save("../results/time.npy", t)

# Power and SoC plotting (eg, debugging)
fig, axs = plt.subplots(2, 1, figsize=(plts["fig_width"], 1.5*plts["fig_width"]), sharex=True)
for i,(field, y_label) in enumerate([("power", "$p_i(t)$ [kW]"), ("soc", "$b_i(t)$ [kWh]")]):
    axs[i].plot(t, x_eg[field][:,:ret_i], color="k", linewidth=0.3)
    axs[i].set_ylabel(y_label)
    axs[i].grid(alpha=0.2)
axs[-1].set_xlabel("Time (h)")
plt.show(block=False)

fig, axs = plt.subplots(2, 1, figsize=(plts["fig_width"], plts["fig_width"]), sharex=True)
for i,(field, y_label) in enumerate([("power", "$p_i(t)$ [kW]"), ("soc", "$b_i(t)$ [kWh]")]):
    axs[i].plot(t, x_tr[field][:,:ret_i], color="k", linewidth=0.3)
    axs[i].set_ylabel(y_label)
    axs[i].grid(alpha=0.2)
axs[-1].set_xlabel("Time (h)")
plt.show(block=False)


#%% Plotting
# Collective charging
fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.5*plts["fig_width"]))
ax.plot(t, 
    np.linalg.norm(x_eg["max_soc"] - x_eg["soc"], axis=1)**2 /\
    np.linalg.norm(x_eg["max_soc"] - x_eg["soc"][0,:], axis=1)**2,
    label=r"$\displaystyle \frac{\left\| \overline{\mathbf{b}} - \mathbf{b}(t) \right\|^2}" +\
          r"{\left\| \overline{\mathbf{b}} - \mathbf{b}(0) \right\|^2}$"
)
ax.plot(t, 
    np.linalg.norm(x_tr["max_soc"] - x_tr["soc"], axis=1)**2 /\
    np.linalg.norm(x_tr["max_soc"] - x_tr["soc"][0,:], axis=1)**2,
    label=r"$\displaystyle \frac{\left\| \overline{\mathbf{b}}^\dagger - \mathbf{b}^\dagger(t) \right\|^2}" +\
          r"{\left\| \overline{\mathbf{b}}^\dagger - \mathbf{b}^\dagger(0) \right\|^2}$"
)
ax.set_xlabel("Time [h]")
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig("../fig/charging.pdf", bbox_inches="tight")
plt.show(block=False) 
 
# Precedence function
fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.5*plts["fig_width"]))
ax.plot(t, np.apply_along_axis(precedence, 1, x_eg).std(axis=1), label=r"$\text{std}\{\mathbf{g}(t)\}$")
ax.plot(t, np.apply_along_axis(precedence, 1, x_tr).std(axis=1), label=r"$\text{std}\{\mathbf{g}^\dagger(t)\}$")
ax.set_xlabel("Time [h]")
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig("../fig/precedence.pdf", bbox_inches="tight")
plt.show(block=False) 

# Availability and aggregate power
fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.5*plts["fig_width"]))
ax.plot(t, x_eg["power"][:,ret_i], label=r"$p_\ell(t)$")
ax.plot(t, x_tr["power"][:,ret_i], label=r"$p^\dagger_\ell(t)$")
ax.plot(t, x_eg["power"].sum(1), color="k", linestyle="--", linewidth=0.5, 
        label="$\sum_{i \in \mathcal{N}} p_i(t)$")
ax.set_ylabel("Power [kW]")
ax.set_xlabel("Time [h]")
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig("../fig/aggregate_power.pdf", bbox_inches="tight")
plt.show(block=False)

# # # State
# # is_active_view = 1 - y[:,IS_ACTIVE_INDICES]
# # is_active_view[is_active_view == 0] = np.nan
# # axs[4].plot(t, np.arange(NUM_AGENTS) * is_active_view, color="k", linewidth=2)
# # axs[4].set_ylabel("$i \in \mathcal{C}$")

# plt.tight_layout()
# plt.savefig("../fig/results.pdf", bbox_inches="tight")
# plt.show(block=False) 
