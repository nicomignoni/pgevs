import numpy as np
import matplotlib.pyplot as plt

class System:
    fields = [
        ('max_output', '<f4'), ('power', '<f4'), ('max_input', '<f4'), ('max_soc', '<f4'), 
        ('soc', '<f4'), ('alpha', '<f4'), ('epsilon', '<f4'), ('eta', '<f4'), ('gamma', '<f4'), 
        ('mu', '<f4'), ('sigma', '<f4'), ('is_active', '<f4'), ('max_power', '<f4')
    ]

    def __init__(self, init_x, settings, color, var_symbol):
        self.s = settings
        self.color = color
        self.var_symbol = var_symbol
        num_agents = self.s["num_cs"] + 1
        num_steps = int(self.s["t_bound"] / self.s["step_size"])

        self.k = 0
        self.t = self.s["step_size"] * np.arange(num_steps+1) 

        self.x = np.zeros([num_steps+1, num_agents], dtype=System.fields)
        self.x[:][self.k,:] = init_x

        self.params = ['max_power', 'max_soc', 'is_active', 'alpha', 'epsilon', 'eta', 'gamma', 'mu', 'sigma']
        
    def efficiency(self, k): 
        return self.x["eta"][k,:] - self.x["gamma"][k,:] + self.x["gamma"][k,:]*np.tanh(
               self.x["max_soc"][k,:]*self.x["sigma"][k,:] - self.x["epsilon"][k,:]*self.x["soc"][k,:]
        )

    def degradation(self, k): 
        return self.x["mu"][k,:] * self.x["power"][k,:]

    def energy_cost(self, k): 
        return self.s["energy_price"] * self.x["power"][k,:-1].sum() * self.s["step_size"]

    def emptyness(self, k): # 0 / 0 -> 0
        return np.divide(
            self.x["max_soc"][k,:] - self.x["soc"][k,:],
            self.x["max_soc"][k,:],
            where=self.x["max_soc"][k,:]!=0
        )

    def precedence(self, k): 
        return self.x["alpha"][k,:] * self.emptyness(k) -\
               (1 - self.x["alpha"][k,:]) * self.degradation(k) 

    def rho(self, k):
        g = self.precedence(k) 
        return self.x["is_active"][k,:] *\
               self.x["is_active"][k,:,None] *\
               np.maximum(0, self.x["max_power"][k,:,None] - self.x["power"][k,:,None])*\
               np.maximum(0, g[:,None] - g)

    def update_css_status(self, k):
        self.x["is_active"][k,:-1] = np.logical_and(
            self.x["max_soc"][k,:-1] > 0, 
            self.x["soc"][k,:-1] < self.x["max_soc"][k,:-1]
        )

    def has_any_ev_departed(self):
        return np.logical_and(
            self.x["max_soc"][:-1,:] > 0, 
            self.x["max_soc"][1:,:] == 0
        ).any(1)

    def has_any_ev_arrived(self):
        return np.logical_and(
            self.x["max_soc"][:-1,:] == 0, 
            self.x["max_soc"][1:,:] > 0
        ).any(1)

    @staticmethod
    def uniform_power_allocation(x, settings):
        power = np.zeros(settings["num_cs"]+1)
        power[-1] = settings["avail"]
        sorted_indices = np.argsort(x["max_power"][:-1])
        for k,i in enumerate(sorted_indices):
            mean = power[-1] / (power[:-1].size - k)
            power[i] = np.minimum(x["max_power"][i] * x["is_active"][i], mean)
            power[-1] -= power[i]
        return power

    # Plotting utilities
    def plot_trajectory(self, **params):
        # Debugging, plot the power and soc trajectories for EVs only
        fig, axs = plt.subplots(2, 1, figsize=params["figsize"], sharex=True)
        for i,(field, y_label) in enumerate([("power", "$p_i(t)$ [kW]"), ("soc", "$b_i(t)$ [kWh]")]):
            axs[i].plot(self.t, self.x[field][:,:-1], color="k", linewidth=0.3)
            axs[i].set_ylabel(y_label)
            axs[i].grid(alpha=0.2)
        axs[-1].set_xlabel("Time (h)")
        plt.show(block=False) 

    @staticmethod
    def plot_collective_charging(systems, **params):
        fig = plt.figure("Collective charging", figsize=params["figsize"])
        ax = fig.add_subplot()
        for system in systems:
            ax.plot( 
                system.t, 100 * system.collective_charging, 
                label=r"$\phi" + f"{system.var_symbol}(t)$",
                color=system.color, 
                linewidth=0.9
            )
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("[\%]")
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=4)
        ax.grid()
        plt.show(block=False)

    @staticmethod
    def plot_fairness(systems, **params):
        fig = plt.figure("Fairness", figsize=params["figsize"])
        ax = fig.add_subplot()
        for system in systems:
            ax.plot(
                system.t, system.fairness, 
                label=r"$\text{std}(\mathbf{g}" + f"{system.var_symbol}(t))$",
                color=system.color, 
                linewidth=0.9
            )
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("€")
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=4)
        ax.grid()
        plt.tight_layout()
        plt.show(block=False) 

    @staticmethod
    def plot_availability_use(systems, **params):
        fig = plt.figure("Availability", figsize=params["figsize"])
        ax = fig.add_subplot()
        for system in systems:
            ax.plot(
                system.t, 100 * system.availability_use, 
                label=r"$\sum_{i \in \mathcal{C}} p_i" + f"{system.var_symbol}(t)$", 
                color=system.color, 
                linewidth=0.9
            )
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("[\%]")
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=4)
        ax.grid()
        plt.show(block=False)
    
    # KPIs and metrics
    @property
    def fairness(self):
        return self.precedence(slice(None)).std(axis=1)

    @property
    def collective_charging(self):
        return np.linalg.norm(self.x["max_soc"] - self.x["soc"], axis=1)**2 /\
               np.linalg.norm(self.x["max_soc"] - self.x["soc"][0,:], axis=1)**2

    @property
    def availability_use(self):
        return self.x["power"][:,:-1].sum(1) / self.s["avail"]
    

class EvolutionaryDynamic(System):
    def __init__(self, init_x, settings, color="tab:blue", var_symbol=""):
        super().__init__(init_x, settings, color, var_symbol)
        self.name = self.__class__.__name__

    def make_flow_step(self):
        # Jump (full SoC, optimal power reallocation)
        self.update_css_status(self.k)
        reallocable_power = np.sum(self.x["power"][self.k,:] * (1 - self.x["is_active"][self.k,:]))
        reallocation_order = np.argsort(self.precedence(self.k))[::-1]
        max_power_increase = self.x["is_active"][self.k,:] * (self.x["max_power"][self.k,:] - self.x["power"][self.k,:])
        cum_power_increase = np.cumsum(max_power_increase[reallocation_order])
        full_increase_evs = cum_power_increase <= reallocable_power
        first_non_full_ev = np.argmin(full_increase_evs)
        self.x["power"][self.k,reallocation_order] += full_increase_evs * max_power_increase[reallocation_order]
        self.x["power"][self.k,reallocation_order[first_non_full_ev]] += \
            reallocable_power - max_power_increase[reallocation_order[:first_non_full_ev]].sum()
        self.x["power"][self.k,:] *= self.x["is_active"][self.k,:]

        # Flow 
        r = self.x["power"][self.k,:] * self.rho(self.k)
        self.x["power"][self.k+1,:] = self.x["power"][self.k,:] + (np.sum(r,1) - np.sum(r,0)) * self.s["step_size"] 
        self.x["soc"][self.k+1,:] = self.x["soc"][self.k,:] + (self.efficiency(self.k) * self.x["power"][self.k,:]) * self.s["step_size"] 

        # Broadcast the remaining fields to the next time step
        self.x[self.params][self.k+1,:] = np.copy(self.x[self.params][self.k,:])

        self.k += 1


class UniformDynamic(System):
    def __init__(self, init_x, settings, color="tab:orange", var_symbol="^\dagger"):
        super().__init__(init_x, settings, color, var_symbol)

    def make_flow_step(self):
        num_evs, num_active_evs = np.sum(self.x["max_soc"][self.k,:] > 0), np.sum(self.x["is_active"][self.k,:])
        self.update_css_status(self.k)
        new_num_evs, new_num_active_evs = np.sum(self.x["max_soc"][self.k,:] > 0), np.sum(self.x["is_active"][self.k,:])

        if new_num_evs != num_evs or new_num_active_evs != num_active_evs:
            self.x["power"][self.k+1,:] = System.uniform_power_allocation(self.x[:][self.k,:], self.s)
        else:
            self.x["power"][self.k+1,:] = np.copy(self.x["power"][self.k,:])
        self.x["soc"][self.k+1,:] = self.x["soc"][self.k,:] + (self.efficiency(self.k) * self.x["power"][self.k,:]) * self.s["step_size"] 

        # Broadcast the remaining fields to the next time step
        self.x[self.params][self.k+1,:] = np.copy(self.x[self.params][self.k,:])

        self.k += 1


