import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt

SECONDS_IN_HOUR = 3600

# Figure settings
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
def precedence(x): # 0 / 0 -> 0
    return params["rush"] * np.divide(
        params["max_soc"] - x[SOC_INDICES],
        params["max_soc"], 
        out=np.zeros_like(x[SOC_INDICES]), 
        where=params["max_soc"]!=0
    )

def rho(x):
    g = precedence(x) 
    return x[IS_ACTIVE_INDICES] * x[IS_ACTIVE_INDICES, None]*\
           np.maximum(0, params["max_net_power"][:,None] - x[POWER_INDICES,None])*\
           np.maximum(0, g[:,None] - g)

        
def dx(t, x): # Parameters are taken as globals
    r = x[POWER_INDICES] * rho(x)
    return np.hstack([
        np.sum(r,1) - np.sum(r,0), # power dynamics
        params["efficiency"] * x[POWER_INDICES], # state-of-charge dynamics
        np.zeros(NUM_AGENTS) # is_active dynamics, i.e. trivially null
    ])

# System settings
NUM_CHARGING_STATION = 20
NUM_AGENTS = NUM_CHARGING_STATION + 1
MIN_EFFICIENCY = 0.98
INIT_SOC_FRAC = 0.5
AVAILABILITY = 200

# Load data and add rush and efficiency columns for the EVs data
EVS_DF = pd.read_csv("data/EVs.csv")
CSS_DF = pd.read_csv("data/CSs.csv")
# EVS_DF["max_soc"] *= SECONDS_IN_HOUR # kWh -> kWs 

# Simulation settings
T_BOUND = 10
MIN_STEP = 1e-3
MAX_STEP = MIN_STEP
ABS_TOL = 1e-6
REL_TOL = ABS_TOL

# Variables indexing
RETAILER_POWER_INDEX = NUM_CHARGING_STATION
POWER_INDICES        = np.arange(NUM_AGENTS)
SOC_INDICES          = np.arange(NUM_AGENTS, 2*NUM_AGENTS)
IS_ACTIVE_INDICES    = np.arange(2*NUM_AGENTS, 3*NUM_AGENTS) 

# Sample CSs and PEVs, add necessary columns, and create paramteres
params_df = pd.concat((
        EVS_DF.sample(NUM_CHARGING_STATION, replace=True, ignore_index=True),
        CSS_DF.sample(NUM_CHARGING_STATION, replace=True, ignore_index=True),
        pd.DataFrame({"rush": np.random.rand(NUM_CHARGING_STATION), 
                      "efficiency": np.random.uniform(low=MIN_EFFICIENCY, size=NUM_CHARGING_STATION)})
    ),
    axis=1
)
params_df = pd.concat((
    params_df, 
    pd.DataFrame({"ev_name": ["Empty"], "max_input_power": [np.inf], 
                  "max_soc": [0], "rush": [0], "efficiency": [0],
                  "css_type": ["Retailer"], "max_output_power": [AVAILABILITY]}),
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
system = sp.integrate.RK23(
    dx, y0=init_x, t0=0, t_bound=T_BOUND, 
    first_step=MIN_STEP, max_step=MAX_STEP, rtol=REL_TOL, atol=ABS_TOL
)
y, t = [], []
while system.status != "finished":
    system.step()
    y.append(system.y)
    t.append(system.t)

    # Jumps
    system.y[IS_ACTIVE_INDICES] = (system.y[SOC_INDICES] <= params["max_soc"])
    system.y[RETAILER_POWER_INDEX] += np.sum(system.y[POWER_INDICES] * (1 - system.y[IS_ACTIVE_INDICES]))
    system.y[POWER_INDICES] *= system.y[IS_ACTIVE_INDICES]

# Plotting
Y = np.array(y)
fig, axs = plt.subplots(4, 1, figsize=(6.4,6), sharex=True)
axs[-1].set_xlabel("Time (s)")

for i,(indices, y_label) in enumerate((
    (POWER_INDICES, "$p_i(t)$ [kW]"), (SOC_INDICES, "$b_i(t)$ [kWh]"), (IS_ACTIVE_INDICES, "$s_i(t)$"))
):
    axs[i].plot(t, Y[:,indices], color="#333333", linewidth=0.5)
    axs[i].set_ylabel(y_label)
    axs[i].grid(alpha=0.2)

axs[-1].plot(t, Y[:,POWER_INDICES].sum(1), color="#333333", linewidth=0.5)
axs[-1].set_ylabel(r"$\sum_{i \in \mathcal{N}} p_i(t)$ [kW]")
axs[-1].grid(alpha=0.2)

plt.show(block=False) 