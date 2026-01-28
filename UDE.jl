using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, CSV, DataFrames, DataInterpolations, Statistics

# Comment out either mouse or human, as desired

# --- MOUSE DATA PREPROCESSING ---

data = CSV.read("/data/mouse_data.txt", DataFrame, delim='\t')

# Extract data for a single subject (set row.ID)
data_id = filter(row -> row.ID == 1, data)
t_exp = data_id.Time          
volume_exp = data_id.Observation 

# Sigmoid function
function sigmoid(x; steepness=1.0)
    return 1 / (1 + exp(-steepness * x))
end

# Sigmoid interpolation function
function sigmoid_interpolation(t, a, b; steepness=2.0)
    t_min, t_max = minimum(t_exp), maximum(t_exp)
    t_scaled = (t - t_min) / (t_max - t_min)
    x = (t_scaled - 0.5) * 12  
    s = sigmoid(x; steepness=steepness)
    return a * (1 - s) + b * s
end

# Extract start & end values
a = volume_exp[1]
b = volume_exp[end]
# Range of t
t_values = t_exp[1]:0.1:t_exp[end]
# Perform interpolation
interpolation = [sigmoid_interpolation(t, a, b) for t in t_values]

# Plot the experimental data with the interpolated data
scatter(t_exp, volume_exp, label="Experimental Data", color=:orange, markersize = 6)
plot!(t_values, interpolation, label="Sigmoid Interpolation", color=:purple, lw = 3)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")


# --- HUMAN DATA PREPROCESSING ---

df = CSV.read("/data/preprocessed_tumor_data.csv", DataFrame)

# Select study and patient id
study = "Study1"
patient_id = 3

study_df = filter(row -> row.study == study, df)
patient_df = filter(row -> row.patient_id == patient_id, study_df)

t_exp = Float64.(patient_df.time)
volume_exp = Float64.(patient_df.volume_mm3)

# Akima Interpolation 
interp_akima = AkimaInterpolation(volume_exp, t_exp)

# Generate smooth time points for plotting
t_values = range(t_exp[1], t_exp[end], length=200)

# Evaluate interpolation
interpolation = interp_akima.(t_values)

# Plot the experimental data with the interpolated data
p = scatter(t_exp, volume_exp, label="Experimental Data", color=:orange, markersize=8, markerstrokewidth=2)
plot!(p, t_values, interpolation, label="Akima Interpolation", color=:purple, lw=2, linestyle=:dash)
xlabel!(p, "Time (days)")
ylabel!(p, "Tumor Volume (mm³)")
plot!(p, legend=:topleft)

# --- UDE ---

rng = Xoshiro(123)

# Normalization
V_mean = mean(interpolation)
V_std = std(interpolation)

t_min = minimum(t_values)
t_max = maximum(t_values)

# NN that replaces the a term
NN1 = Lux.Chain(
    Lux.Dense(2, 10, tanh), 
    Lux.Dense(10, 10, tanh), 
    Lux.Dense(10, 1)
)
p1, st1 = Lux.setup(rng, NN1)

# NN that replaces the log term
NN2 = Lux.Chain(
    Lux.Dense(2, 10, tanh), 
    Lux.Dense(10, 10, tanh), 
    Lux.Dense(10, 1)
)
p2, st2 = Lux.setup(rng, NN2)

p0_vec = ComponentArray(layer_1=p1, layer_2=p2)

# Define the UDE function
function ude_func!(du, u, p, t)
    V = u[1]

    # Normalize inputs
    V_norm = (V - V_mean) / V_std
    t_norm = (t - t_min) / (t_max - t_min)

    inputs = reshape([V_norm, t_norm], :, 1) 

    NN_a_output, _ = NN1(inputs, p.layer_1, st1)
    NN_log_output, _ = NN2(inputs, p.layer_2, st2)

    NN_a = log1p(exp(NN_a_output[1]))
    NN_log = NN_log_output[1]

    du[1] = NN_a * V * NN_log  
end

u0_raw = [interpolation[1]]
tspan_raw = (minimum(t_values), maximum(t_values))

# Prediction function
function predict_ude(p; solver = Tsit5())
    prob = ODEProblem(ude_func!, u0_raw, tspan_raw, p)
    sol = solve(prob, solver, saveat=collect(t_values))
    return Array(sol)[1, :]
end

# Loss function
function loss_ude(p)
    pred = predict_ude(p)
    data_loss = mean(abs2, interpolation .- pred)
    if length(pred) > 1
        pred_derivs = diff(pred)
        true_derivs = diff(interpolation)
        deriv_loss = mean(abs2, true_derivs .- pred_derivs)
        smooth_loss = length(pred_derivs) > 1 ? 0.01 * mean(abs2, diff(pred_derivs)) : 0.0
        total_loss = data_loss + 0.5 * deriv_loss + 0.05 * smooth_loss
    else
        total_loss = data_loss
    end
    return total_loss
end

# Callback function
callback = function (state, l; doplot = true)
    println(l)
    if doplot
        try
            p = state.u
            pred = predict_ude(p)
            if length(pred) == length(interpolation)
                plt = scatter(t_values, interpolation, label="Training Data", color=:mediumpurple, markersize=3)
                plot!(plt, t_values, pred, label="UDE", color=:red, lw=2)
                xlabel!("Time (days)")
                ylabel!("Tumor Volume (mm³)")
                title!("UDE Prediction Vs Training Data")
                display(plt)
            end
        catch e
            println("Error in callback plotting: ", e)
        end
    end
    return false
end

# Optimization
pinit = p0_vec
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, _) -> loss_ude(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback=callback, maxiters=1000)

optprob2 = Optimization.OptimizationProblem(optf, result1.u)
result2 = Optimization.solve(optprob2, OptimizationOptimisers.Adam(0.005), callback=callback, maxiters=1000)

optprob3 = Optimization.OptimizationProblem(optf, result2.u)
result_final = Optimization.solve(optprob3, OptimizationOptimisers.Adam(0.001), callback=callback, maxiters=500)

final_pred = predict_ude(result_final.u)

# Plot the UDE solution
plt = scatter(t_values, interpolation, label="Training Data", color=:mediumpurple, markersize = 3)
plot!(plt, t_values, final_pred, label="UDE Prediction", color=:red, lw=3)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
display(plt)
