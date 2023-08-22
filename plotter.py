import numpy as np
import matplotlib.pyplot as plt

class Plotter:

    @staticmethod
    

    @staticmethod
    

    @staticmethod
    def fairness(t, ax, systems):
        for system in systems:
            ax.plot(
                t, system.fairness, 
                label=r"$\text{std}(\mathbf{g}" + f"{system.var_symbol}(t))$",
                color=system.color, 
                linewidth=0.9
            )
