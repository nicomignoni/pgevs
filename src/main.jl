using Random, Revise, Distributions
using DifferentialEquations, DifferentialEquations.JumpProcesses
using Makie

includet("Simulation.jl")
using .Simulation

Random.seed!(2026)

params = let
    max_buffer_capacity = 100.0
    buffer_at_arrival_distribution = Uniform(0.5, 0.99max_buffer_capacity)
    buffer_intake_capacity_distribution = Uniform(1.0, 10.0)

    num_feeders = 10
    threshold = 0.1
    is_connected_jump_rate = 0.0
    availability = 100.0
    fitness_weight = [1.0; rand(Uniform(0.98, 1.0), num_feeders-1)]
    energy_price = rand(Uniform(1.0, 10.0))
    degradation_coeff = [0.0; rand(Uniform(1.0, 2.0), num_feeders-1)]
    min_efficiency = [0.0; rand(Uniform(0.5, 1.0), num_feeders-1)]
    efficiency_slope = [0.0; rand.(Uniform(1.0, 2.0), num_feeders-1)]
    feeder_max_output = [availability; rand([7.0, 22.0, 44.0, 100.0], num_feeders-1)]

    Simulation.Params(
        num_feeders,
        threshold,
        is_connected_jump_rate,
        availability,
        fitness_weight,
        energy_price,
        degradation_coeff,
        min_efficiency,
        efficiency_slope,
        feeder_max_output,
        max_buffer_capacity,
        buffer_at_arrival_distribution,
        buffer_intake_capacity_distribution,
    )
end

u₀ = Simulation.initial_state(params)
tspan = (0.0, 100.0)

sol_population = Simulation.solve_population(u₀, params, tspan)
sol_uniform = Simulation.solve_alternative(
    Simulation.uniform_commodity_update!,
    u₀,
    params,
    tspan
)
sol_threshold = Simulation.solve_alternative(
    Simulation.threshold_commodity_update!,
    u₀,
    params,
    tspan
)
sol_priority = Simulation.solve_alternative(
    Simulation.priority_commodity_update!,
    u₀,
    params,
    tspan
)
sol_proportional = Simulation.solve_alternative(
    Simulation.proportional_commodity_update!,
    u₀,
    params,
    tspan
)

Simulation.plot_metrics(
    params,
    (sol_population, sol_uniform, sol_threshold, sol_proportional, sol_priority),
    ("pop", "unif", "thres", "prop", "prior"),
)
