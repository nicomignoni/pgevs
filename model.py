import numpy as np


class System:
    fields = [
        ('max_output', '<f8'), ('power', '<f8'), ('max_input', '<f8'), ('max_soc', '<f8'), 
        ('soc', '<f8'), ('alpha', '<f8'), ('epsilon', '<f8'), ('eta', '<f8'), ('gamma', '<f8'), 
        ('mu', '<f8'), ('sigma', '<f8'), ('is_active', '<f8'), ('max_power', '<f8')
    ]

    def __init__(self, init_x, settings):
        self.s = settings
        num_agents = self.s["num_cs"] + 1
        num_steps = int(self.s["t_bound"] / self.s["step_size"])

        self.k = 0
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

    # KPIs and metrics
    def collective_charging(self):
        return np.linalg.norm(self.x["max_soc"] - self.x["soc"], axis=1)**2 /\
               np.linalg.norm(self.x["max_soc"] - self.x["soc"][0,:], axis=1)**2

    def availability_utilization(self):
        return self.x["power"][:,:-1].sum(1) / self.s["avail"]
    

class EvolutionaryDynamic(System):
    def __init__(self, init_x, settings, color="tab:blue", var_symbol=""):
        super().__init__(init_x, settings)
        self.color = color
        self.var_symbol = var_symbol

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
        super().__init__(init_x, settings)
        self.color = color
        self.var_symbol = var_symbol

    def make_flow_step(self):
        self.update_css_status(self.k)

        uniform_allocation = np.divide(self.s["avail"], 
            self.x["is_active"][self.k,:-1].sum(),
            where=self.x["is_active"][self.k,:-1]==1,
            out=np.zeros(self.s["num_cs"])
        ) 
        self.x["power"][self.k+1,:-1] = np.minimum(uniform_allocation, self.x["max_power"][self.k,:-1])
        self.x["power"][self.k+1,-1] = self.s["avail"] - self.x["power"][self.k+1,:-1].sum()
        self.x["soc"][self.k+1,:] = self.x["soc"][self.k,:] + (self.efficiency(self.k) * self.x["power"][self.k,:]) * self.s["step_size"] 
        
        # Broadcast the remaining fields to the next time step
        self.x[self.params][self.k+1,:] = np.copy(self.x[self.params][self.k,:])

        self.k += 1


