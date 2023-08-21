# # Events and handlers
# def after_departures(x): 
#     has_departed = np.random.uniform(size=num_agents) <= s["dep_prob"]
#     for field in ("soc", "max_soc", "max_power", "is_active"): x[field] *= ~has_departed
#     return x


# def after_charge_simple(x):
#     x["is_active"][:-1] = np.logical_and(x["max_soc"][:-1] > 0, x["soc"][:-1] <= x["max_soc"][:-1])
#     x["power"][ri] += np.sum(x["power"][:-1] * (1 - x["is_active"][:-1]))
#     x["power"] *= x["is_active"]
#     return x

# def after_arrivals(x):
#     has_arrived = np.logical_and(x["is_active"] == 0, np.random.uniform(size=num_agents) <= s["arr_prob"])
#     num_arrived_evs = np.count_nonzero(has_arrived)
#     if num_arrived_evs >= 1:
#         new_evs = evs_df.sample(num_arrived_evs, replace=True, ignore_index=True).to_records(index=False)
#         for field in new_evs.dtype.names:
#             x[field][has_arrived] = new_evs[field]
#         x["max_power"] = np.minimum(x["max_input"], x["max_output"])
#     return x



# def dynamic_tr(x):
#     """Jump dynamic"""
#     # x = after_departures(x)
#     x = after_charge_simple(x)
#     # x = after_arrivals(x)

#     uniform_allocation = np.divide(s["avail"], 
#         x["is_active"][:-1].sum(),
#         where=x["is_active"][:-1]==1,
#         out=np.zeros(s["num_cs"])
#     ) 
#     x["power"][ri] = 0
#     x["power"][:-1] = np.minimum(uniform_allocation, x["max_power"][:-1])
#     x["power"][ri] = s["avail"] - x["power"][:-1].sum()
#     x["soc"] += (efficiency(x) * x["power"]) * s["step_size"] 
#     return x


# # CSs data are invariant with respect to any dynamic


# # Save results
# # timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# np.save("../results/output_eg.npy", x_eg)
# np.save("../results/output_tr.npy", x_tr)
# np.save("../results/time.npy", t)

# # Power and SoC plotting (eg, debugging)



# #%% Plotting
# # Collective charging
# fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.5*plts["fig_width"]))
# ax.plot(t, 
#     np.linalg.norm(x_eg["max_soc"] - x_eg["soc"], axis=1)**2 /\
#     np.linalg.norm(x_eg["max_soc"] - x_eg["soc"][0,:], axis=1)**2,
#     label=r"$\displaystyle \frac{\left\| \overline{\mathbf{b}} - \mathbf{b}(t) \right\|^2}" +\
#           r"{\left\| \overline{\mathbf{b}} - \mathbf{b}(0) \right\|^2}$"
# )
# ax.plot(t, 
#     np.linalg.norm(x_tr["max_soc"] - x_tr["soc"], axis=1)**2 /\
#     np.linalg.norm(x_tr["max_soc"] - x_tr["soc"][0,:], axis=1)**2,
#     label=r"$\displaystyle \frac{\left\| \overline{\mathbf{b}}^\dagger - \mathbf{b}^\dagger(t) \right\|^2}" +\
#           r"{\left\| \overline{\mathbf{b}}^\dagger - \mathbf{b}^\dagger(0) \right\|^2}$"
# )
# ax.set_xlabel("Time [h]")
# ax.grid(alpha=0.2)
# ax.legend()
# plt.tight_layout()
# plt.savefig("../fig/charging.pdf", bbox_inches="tight")
# plt.show(block=False) 
 
# # Precedence function
# fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.5*plts["fig_width"]))
# ax.plot(t, np.apply_along_axis(precedence, 1, x_eg).std(axis=1), label=r"$\text{std}\{\mathbf{g}(t)\}$")
# ax.plot(t, np.apply_along_axis(precedence, 1, x_tr).std(axis=1), label=r"$\text{std}\{\mathbf{g}^\dagger(t)\}$")
# ax.set_xlabel("Time [h]")
# ax.grid(alpha=0.2)
# ax.legend()
# plt.tight_layout()
# plt.savefig("../fig/precedence.pdf", bbox_inches="tight")
# plt.show(block=False) 

# # Availability and aggregate power
# fig, ax = plt.subplots(figsize=(plts["fig_width"], 0.5*plts["fig_width"]))
# ax.plot(t, x_eg["power"][:,ri], label=r"$p_\ell(t)$")
# ax.plot(t, x_tr["power"][:,ri], label=r"$p^\dagger_\ell(t)$")
# ax.plot(t, x_eg["power"].sum(1), color="k", linestyle="--", linewidth=0.5, 
#         label="$\sum_{i \in \mathcal{N}} p_i(t)$")
# ax.set_ylabel("Power [kW]")
# ax.set_xlabel("Time [h]")
# ax.grid(alpha=0.2)
# ax.legend()
# plt.tight_layout()
# plt.savefig("../fig/aggregate_power.pdf", bbox_inches="tight")
# plt.show(block=False)

# # # # State
# # # is_active_view = 1 - y[:,IS_ACTIVE_INDICES]
# # # is_active_view[is_active_view == 0] = np.nan
# # # axs[4].plot(t, np.arange(NUM_AGENTS) * is_active_view, color="k", linewidth=2)
# # # axs[4].set_ylabel("$i \in \mathcal{C}$")

# # plt.tight_layout()
# # plt.savefig("../fig/results.pdf", bbox_inches="tight")
# # plt.show(block=False) 
