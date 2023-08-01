from datetime import datetime
from itertools import count

import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt

np.random.seed(2023)

SECONDS_IN_HOUR = 3600

# Figure settings
FIG_WIDTH = 3.5  
LATEX_PREAMBLE = [
    r'\usepackage{amsfonts}',
    r'\usepackage{amssymb}',
    r'\usepackage{amsmath}',
]

plt.rcParams.update({
    "grid.alpha": 0.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "ytick.labelsize": 6,
    "xtick.labelsize": 6,
    "font.size": 8,
    "font.family": "sans",
    "font.sans-serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "text.latex.preamble": "".join(LATEX_PREAMBLE)
})

# Dynamics 
def efficiency(x): 
    return params["gamma"] * np.tanh(params["max_soc"] * params["sigma"] - x[SOC_INDICES]) +\
           params["eta"] - params["gamma"]
def degradation(x): return params["mu"] * x[POWER_INDICES]
def energy_cost(x): return params["lambda"] * x[POWER_INDICES[:-1]].sum()
def charge_level(x): # 0 / 0 -> 0
    return np.divide(
        params["max_soc"] - x[SOC_INDICES],
        params["max_soc"], 
        out=np.zeros_like(x[SOC_INDICES]), 
        where=params["max_soc"]!=0
    )

def precedence(x): 
    return params["alpha"]       * charge_level(x) -\
           (1 - params["alpha"]) * (degradation(x) + energy_cost(x))

def rho(x):
    g = precedence(x) 
    return x[IS_ACTIVE_INDICES] * x[IS_ACTIVE_INDICES, None]*\
           np.maximum(0, params["max_net_power"][:,None] - x[POWER_INDICES,None])*\
           np.maximum(0, g[:,None] - g)

        
def dx(t, x): # Parameters are taken as globals
    r = x[POWER_INDICES] * rho(x)
    return np.hstack([
        np.sum(r,1) - np.sum(r,0), # power dynamics
        efficiency(x) * x[POWER_INDICES], # state-of-charge dynamics
        np.zeros(NUM_AGENTS) # is_active dynamics, i.e. trivially null
    ])

# System settings
NUM_CHARGING_STATION = 20
NUM_AGENTS = NUM_CHARGING_STATION + 1

INIT_SOC_FRAC = 0.5
AVAILABILITY = 200

MIN_ETA = 0.98
MIN_ALPHA, MAX_ALPHA = 0.8, 1
MIN_SIGMA, MAX_SIGMA = 0.1, 0.9
MIN_GAMMA, MAX_GAMMA = 0.05, 0.15
MIN_EPSIL, MAX_EPSIL = 0.1, 0.2
MIN_MU, MAX_MU = 0.01, 0.5
MIN_LAMBDA, MAX_LAMBDA = 0.01, 0.05

# Load data and add alpha and efficiency columns for the EVs data
EVS_DF = pd.read_csv("data/EVs.csv")
CSS_DF = pd.read_csv("data/CSs.csv")
# EVS_DF["max_soc"] *= SECONDS_IN_HOUR # kWh -> kWs 

# Simulation settings
T_BOUND = 100
MIN_STEP = MAX_STEP = 1e-3
ABS_TOL = REL_TOL = 1e-6

# Variables indexing
RETAILER_POWER_INDEX = NUM_CHARGING_STATION
POWER_INDICES        = np.arange(NUM_AGENTS)
SOC_INDICES          = np.arange(NUM_AGENTS, 2*NUM_AGENTS)
IS_ACTIVE_INDICES    = np.arange(2*NUM_AGENTS, 3*NUM_AGENTS) 

# Sample CSs and PEVs, add necessary columns, and create paramteres
params_df = pd.concat((
        EVS_DF.sample(NUM_CHARGING_STATION, replace=True, ignore_index=True),
        CSS_DF.sample(NUM_CHARGING_STATION, replace=True, ignore_index=True),
        pd.DataFrame({
            "alpha": np.random.uniform(MIN_ALPHA, MAX_ALPHA, size=NUM_CHARGING_STATION), 
            "eta": np.random.uniform(low=MIN_ETA, size=NUM_CHARGING_STATION),
            "sigma": np.random.uniform(MIN_SIGMA, MAX_SIGMA, size=NUM_CHARGING_STATION),
            "gamma": np.random.uniform(MIN_GAMMA, MAX_GAMMA, size=NUM_CHARGING_STATION),
            "epsil": np.random.uniform(MIN_EPSIL, MAX_EPSIL, size=NUM_CHARGING_STATION),
            "mu": np.random.uniform(MIN_MU, MAX_MU, size=NUM_CHARGING_STATION),
            "lambda": np.random.uniform(MIN_LAMBDA, MAX_LAMBDA) * np.ones(NUM_CHARGING_STATION)
        })
    ),
    axis=1
)
params_df = pd.concat((
    params_df, 
    pd.DataFrame({
        "ev_name": ["Empty"], "max_input_power": [np.inf], "max_soc": [0], "alpha": [0], "css_type": ["Retailer"], 
        "max_output_power": [AVAILABILITY], "eta": [0], "sigma": [0], "gamma": [0], "mu": [0], "lambda": [0], "epsil": [0]
        }),
    )
)
params_df["max_net_power"] = params_df[["max_input_power", "max_output_power"]].min(axis=1)

params = np.array(params_df.to_records(index=False))

# Initial values
init_x = np.hstack([
    np.hstack([np.zeros(NUM_CHARGING_STATION), AVAILABILITY]), # init power
    INIT_SOC_FRAC * params["max_soc"] * np.random.rand(NUM_AGENTS), # init_soc
    (params["max_soc"] > 0) # init is_active
])
init_x[-1] = 1 # the retailer is always active

# System loop
counter = count()
num_steps = int(T_BOUND / MIN_STEP)
y = np.zeros((num_steps, init_x.size), dtype=np.float16)
t = np.zeros((num_steps,), dtype=np.float16)

system = sp.integrate.RK23(
    dx, y0=init_x, t0=0, t_bound=T_BOUND, first_step=MIN_STEP, 
    max_step=MAX_STEP, rtol=REL_TOL, atol=ABS_TOL
)
n = next(counter)
while system.status != "finished" and n < t.size:
    system.step()
    y[n,:] = system.y
    t[n] = system.t

    # Jumps
    system.y[IS_ACTIVE_INDICES] = (system.y[SOC_INDICES] <= params["max_soc"])
    system.y[RETAILER_POWER_INDEX] += np.sum(system.y[POWER_INDICES] * (1 - system.y[IS_ACTIVE_INDICES]))
    system.y[POWER_INDICES] *= system.y[IS_ACTIVE_INDICES]

    n = next(counter)

# Save results
# timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
np.save(f"results/output.npy", y)
np.save(f"results/time.npy", t)

# Plotting
fig, axs = plt.subplots(5, 1, figsize=(FIG_WIDTH, 1.8*FIG_WIDTH), sharex=True)
axs[-1].set_xlabel("Time (s)")

for i,(indices, y_label) in enumerate((
    (POWER_INDICES[:-1], "$p_i(t)$ [kW]"), (SOC_INDICES, "$b_i(t)$ [kWh]"), (IS_ACTIVE_INDICES, "$s_i(t)$"))
):
    axs[i].plot(t, y[:,indices], color="#333333", linewidth=0.5)
    axs[i].set_ylabel(y_label)
    axs[i].grid(alpha=0.2)

# Availability
axs[3].axhline(AVAILABILITY, linestyle="--", color="tab:red", linewidth=0.5, label="$A$")
axs[3].plot(t, y[:,RETAILER_POWER_INDEX], color="tab:blue", linewidth=0.5, label="$p_ell(t)$")
axs[3].plot(t, y[:,POWER_INDICES].sum(1), color="#333333", linewidth=0.5, label="$\sum_{i \in \mathcal{C}} p_i(t)$")
axs[3].set_ylabel("[kW]")
axs[3].grid(alpha=0.2)

axs[4].plot(t, np.apply_along_axis(precedence, 1, y), color="#333333", linewidth=0.5)
axs[4].set_ylabel(r"$g_i(t)$")
axs[4].grid(alpha=0.2)

plt.tight_layout()
plt.show(block=False) 