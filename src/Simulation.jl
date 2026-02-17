module Simulation

# Model -----------------------------------------------------------------------------------
using Dates, Serialization, Distributions, ComponentArrays
using DifferentialEquations, DifferentialEquations.JumpProcesses

struct Params
    num_feeders::Int
    threshold::Float64
    is_connected_jump_rate::Float64
    availability::Float64
    fitness_weight::Vector{Float64}
    energy_price::Float64
    degradation_coeff::Vector{Float64}
    min_efficiency::Vector{Float64}
    efficiency_slope::Vector{Float64}
    feeder_max_output::Vector{Float64}
    max_buffer_capacity::Float64
    buffer_at_arrival_distribution::UnivariateDistribution
    buffer_intake_capacity_distribution::UnivariateDistribution
end

function unit_vector_normalized(x::Vector)
    m, M = minimum(x), maximum(x)
    return (x .- m) ./ (M - m)
end

function initial_state(p::Params)
    is_active = ones(p.num_feeders) # [1.0; rand([0.0, 1.0], p.num_feeders-1)]
    is_connected = ones(p.num_feeders) # is_active .* [1.0; rand([0.0, 1.0], p.num_feeders-1)]

    buffer = is_active .* [0.0; rand(p.buffer_at_arrival_distribution, p.num_feeders-1)]
    buffer_capacity = [0.0; rand.(Uniform.(buffer[2:end], p.max_buffer_capacity))]
    buffer_intake_capacity =
        [p.availability; rand([7.0, 22.0, 44.0, 100.0], p.num_feeders-1)]

    max_output = min.(p.feeder_max_output, buffer_intake_capacity)
    commodity = let v = rand.(Uniform.(0.0, max_output))
        p.availability * v ./ sum(v)
    end

    return ComponentVector(
        commodity = commodity,
        buffer = buffer,
        is_active = is_active,
        is_connected = is_connected,
        buffer_capacity = buffer_capacity,
        buffer_intake_capacity = buffer_intake_capacity,
        max_output = max_output,
    )
end

# Buffer dynamics
function efficiency(u, p::Params)
    return p.min_efficiency .+
           (p.min_efficiency .- 1.0) .*
           tanh.(p.efficiency_slope .* (u.buffer_capacity .- u.buffer))
end

function buffer_update(u, p::Params)
    return u.commodity .* efficiency(u, p)
end

# Jump variables have no flow dynamics
function set_null_flow_jump_variables!(du, p::Params)
    du.is_active[:] = zeros(p.num_feeders)
    du.is_connected[:] = zeros(p.num_feeders)
    du.buffer_capacity[:] = zeros(p.num_feeders)
    du.buffer_intake_capacity[:] = zeros(p.num_feeders)
    du.max_output[:] = zeros(p.num_feeders)
end

# Population dynamics elements
function cost(u, p::Params)
    return u.commodity .* (p.degradation_coeff .+ p.energy_price * sum(u.commodity[2:end]))
end

function fitness(u, p::Params)
    g = p.fitness_weight .* (u.buffer_capacity .- u.buffer) ./ u.buffer .-
        (1.0 .- p.fitness_weight) .* cost(u, p)
    g[1] = 0.0
    return g
end

function transition_rate(u, p::Params)
    g = fitness(u, p)
    return (u.is_active * u.is_active') .* max.(0.0, u.max_output .- u.commodity)' .*
           max.(0.0, g' .- g)
end

function commodity_update_population(u, p::Params)
    ρ = transition_rate(u, p)
    return (ρ' - ρ) * u.commodity
end

function flow_dynamics_population!(du, u, p, t)
    du.commodity = commodity_update_population(u, p)
    du.buffer = buffer_update(u, p)

    set_null_flow_jump_variables!(du, p)
end

# Alterative commodity allocation rules
function flow_dynamics_alteratives!(du, u, p, t)
    du.commodity .= 0.0
    du.buffer = buffer_update(u, p)

    set_null_flow_jump_variables!(du, p)
end

function uniform_commodity_update!(integrator)
    integrator.u.commodity[2:end] = min.(
        integrator.u.is_active[2:end] .* integrator.p.availability /
        sum(integrator.u.is_active[2:end]),
        integrator.u.max_output[2:end],
    )
    integrator.u.commodity[1] =
        integrator.p.availability - sum(integrator.u.commodity[2:end])
end

function threshold_commodity_update!(integrator)
    g = fitness(integrator.u, integrator.p)
    is_eligible =
        Bool.(integrator.u.is_active) .&&
        (unit_vector_normalized(g) .>= integrator.p.threshold)
    is_eligible[1] = false

    integrator.u.commodity =
        is_eligible .*
        min.(integrator.p.availability / sum(is_eligible), integrator.u.max_output)
    integrator.u.commodity[1] =
        integrator.p.availability - sum(integrator.u.commodity[2:end])
end

function priority_commodity_update!(integrator)
    g = fitness(integrator.u, integrator.p)
    priority_queue = sortperm(g[2:end])
    for (idx, i) in enumerate(priority_queue)
        integrator.u.commodity[i] = min(
            integrator.p.availability -
            sum(integrator.u.commodity[priority_queue[1:(idx-1)]]),
            integrator.u.max_output[i],
        )
    end
    integrator.u.commodity[1] =
        integrator.p.availability - sum(integrator.u.commodity[2:end])
end

function proportional_commodity_update!(integrator)
    g_nonneg = max.(0.0, integrator.u.is_active .* fitness(integrator.u, integrator.p))
    proportions = integrator.u.is_active .* g_nonneg ./ sum(g_nonneg)
    proportions[1] = 0.0

    integrator.u.commodity =
        min.(integrator.p.availability .* proportions, integrator.u.max_output)
    integrator.u.commodity[1] =
        integrator.p.availability - sum(integrator.u.commodity[2:end])
end

# Events and triggers
function disconnection_update!(integrator, i)
    integrator.u.commodity[1] += integrator.u.commodity[i] # surplus back to slack feeder
    integrator.u.commodity[i] = 0.0
    integrator.u.is_active[i] = 0.0
    integrator.u.is_connected[i] = 0.0
    integrator.u.buffer_intake_capacity[i] = 0.0
    integrator.u.max_output[i] = 0.0
    # buffer is left alone, just for drawing purposes (to see the buffer reaching 
    # full capacity). In principle, we should also set:
    # integrator.u.buffer[i] = 0.0
    # integrator.u.buffer_capacity[i] = 0.0
end

function is_connected_update!(integrator)
    i = rand(2:integrator.p.num_feeders)
    if integrator.u.is_connected[i] == 1.0 # buffer disconnection
        disconnection_update!(integrator, i)
    else # buffer connection
        integrator.u.commodity[i] = 0.0 # technically, it was 0.0 also before
        integrator.u.buffer[i] = rand(integrator.p.buffer_at_arrival_distribution)
        integrator.u.is_active[i] = 1.0
        integrator.u.is_connected[i] = 1.0
        integrator.u.buffer_capacity[i] =
            rand(Uniform(integrator.u.buffer[i], integrator.p.max_buffer_capacity))
        integrator.u.buffer_intake_capacity[i] =
            rand(integrator.p.buffer_intake_capacity_distribution)
        integrator.u.max_output[i] =
            min(integrator.p.feeder_max_output[i], integrator.u.buffer_intake_capacity[i])
    end
end

function is_connected_jump_rate(u, p, t)
    return p.num_feeders * p.is_connected_jump_rate
end

function recharging_completion(out, u, t, integrator)
    out[1] = 1.0 # conventionally, we dont check the slack feeder (null) buffer
    out[2:end] = u.buffer_capacity[2:end] .- u.buffer[2:end]
end

function buffer_filling_completion(p::Params)
    VectorContinuousCallback(recharging_completion, disconnection_update!, p.num_feeders)
end

# Build and solve
function solve_population(
    u₀::ComponentVector,
    p::Params,
    tspan;
    solver=Tsit5,
    save::Bool=false,
)
    flow_ode = ODEProblem(flow_dynamics_population!, u₀, tspan, p)
    is_connected_jump = ConstantRateJump(is_connected_jump_rate, is_connected_update!)
    hybrid_ode = JumpProblem(flow_ode, Direct(), is_connected_jump)

    sol = solve(hybrid_ode, solver(); callback = buffer_filling_completion(p))

    if save
        serialize("results/population-$(now()).jld", (params, sol))
    end
    return sol
end

function solve_alternative(
    alternative_commodity_update!,
    u₀::ComponentVector,
    p::Params,
    tspan;
    solver=Rodas5,
    save::Bool=false,
    alterative_approach_name="alternative",
)
    flow_ode = ODEProblem(flow_dynamics_alteratives!, u₀, tspan, p)
    is_connected_jump = ConstantRateJump(is_connected_jump_rate, is_connected_update!)
    hybrid_ode = JumpProblem(flow_ode, Direct(), is_connected_jump)

    cbs = CallbackSet(
        buffer_filling_completion(p),
        PeriodicCallback(alternative_commodity_update!, 1),
    )
    sol = solve(hybrid_ode, solver(); callback = cbs)

    if save
        serialize("results/$(alterative_approach_name)-$(now()).jld", (params, sol))
    end
    return sol
end

# Plotting --------------------------------------------------------------------------------
using Statistics, LinearAlgebra
using Makie, CairoMakie, LaTeXStrings

const INCHES = 96 # pixels

THEME = Theme(
    Axis = (
        xticklabelsize = 6,
        yticklabelsize = 6,
        xlabelsize = 10,
        ylabelsize = 9,
        titlefont = :regular,
        titlesize = 10,
        subtitlesize = 10,
        palette = (color = :tab10,),
        colormap = :tab10,
        colorrange = (1, 10),
    ),
)

set_theme!(merge(theme_latexfonts(), THEME))

function plot_dynamics(sol::ODESolution, p::Params)
    fig = Makie.Figure(size = (3.5INCHES, 1.75INCHES))

    size = (p.num_feeders, length(sol.t))
    buffer = Matrix{Float64}(undef, size)
    buffer_capacity = Matrix{Float64}(undef, size)
    commodity = Matrix{Float64}(undef, size)
    max_output = Matrix{Float64}(undef, size)

    for (t, _) in enumerate(sol.t)
        buffer[:, t] = sol.u[t].buffer
        buffer_capacity[:, t] = sol.u[t].buffer_capacity
        commodity[:, t] = sol.u[t].commodity
        max_output[:, t] = sol.u[t].max_output
    end

    # Plot buffers
    ax_buffer = Makie.Axis(fig[1, 1], xlabel=L"$t$", ylabel = L"$b_i(t)$")
    @assert all(buffer[1, :] .== 0)
    for i = 2:p.num_feeders
        lines!(
            ax_buffer,
            sol.t,
            buffer_capacity[i, :];
            color = :lightgray,
            linestyle = :dash,
            linewidth = 1,
        )
        lines!(ax_buffer, sol.t, buffer[i, :]; color = 1, linewidth = 1)
    end

    # Plot commodity
    ax_commodity = Makie.Axis(fig[1, 2], xlabel=L"$t$", ylabel=L"$p_i(t)$")
    for i = 2:p.num_feeders
        lines!(ax_commodity, sol.t, commodity[i, :]; color = 1, linewidth = 1)
    end

    display(fig)
end

function plot_metrics(p::Params, sols, names; save_fig::Bool = true)
    fig = Makie.Figure(size = (3.5INCHES, 3.5INCHES), figure_padding = 10)
    main = fig[1, 1] = GridLayout()

    filling_plot = Makie.Axis(
        main[1, 1],
        yscale=Makie.pseudolog10,
        ylabel = L"$||\mathbf{B} - \mathbf{b}(t)|| / ||\mathbf{B} - \mathbf{b}(0)||$",
        title = L"\text{a) Overall filling status}",
    )
    dispersion_plot = Makie.Axis(
        main[1, 2],
        yscale=Makie.pseudolog10,
        ylabel = L"$\text{std }\pi(t)$",
        title = L"\text{b) Fitness dispersion}",
    )
    total_plot = Makie.Axis(
        main[2, 1],
        yscale=Makie.pseudolog10,
        xlabel = L"t",
        ylabel = L"$\mathbf{1}^\top \pi(t)$",
        title = L"\text{c) Total fitness}",
    )
    usage_plot = Makie.Axis(
        main[2, 2],
        yscale=Makie.pseudolog10,
        xlabel = L"t",
        ylabel = L"$(\mathbf{1}^\top \mathbf{p}(t) - p_{\ell}(t)) / A$",
        title = L"\text{d) Commodity usage}",
    )

    for (i, (sol, name)) in enumerate(zip(sols, names))
        size = (length(sol.t))
        filling_rate = Vector{Float64}(undef, length(sol.t))
        fitness_std = Vector{Float64}(undef, length(sol.t))
        fitness_total = Vector{Float64}(undef, length(sol.t))
        commodity_usage = Vector{Float64}(undef, length(sol.t))
        for (t, _) in enumerate(sol.t)
            filling_rate[t] =
                norm(sol.u[t].buffer_capacity .- sol.u[t].buffer) /
                norm(sol.u[1].buffer_capacity .- sol.u[1].buffer)
            fitness_std[t] = std(fitness(sol.u[t], p))
            fitness_total[t] = sum(fitness(sol.u[t], p))
            commodity_usage[t] = sum(sol.u[t].commodity[2:end] / p.availability)
        end

        options(i) = (
            alpha = i == 1 ? 1.0 : 0.4,
            color = i,
            linewidth = i == 1 ? 1.2 : 1.0,
            colormap = :tab10,
            colorrange = (1, 10),
            label = name,
        )
        lines!(filling_plot, sol.t, filling_rate; options(i)...)
        lines!(dispersion_plot, sol.t, fitness_std; options(i)...)
        lines!(total_plot, sol.t, fitness_total; options(i)...)
        lines!(usage_plot, sol.t, commodity_usage; options(i)...)
    end

    Legend(
        main[3, 1:2],
        filling_plot,
        orientation = :horizontal,
        framevisible = false,
        labelsize = 8,
        position = :lt,
    )
    rowsize!(main, 3, Relative(0.01))

    save("figures/metrics-$(now()).pdf", fig)
    display(fig)
end

end # module Simulation
