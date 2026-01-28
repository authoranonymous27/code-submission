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


# --- NEURAL ODE ---

# Set up
rng = Xoshiro(123)
dudt_nn = Lux.Chain(
    Lux.Dense(2, 128, tanh), 
    Lux.Dense(128, 128, tanh), 
    Lux.Dense(128, 64, tanh), 
    Lux.Dense(64, 1)
) 
p, st = Lux.setup(rng, dudt_nn)

# Define the neural ODE function
function neural_ode_func!(du, u, p, t)
    inputs = vcat(u, [t])
    y_pred, _ = dudt_nn(inputs, p, st)
    du[1] = y_pred[1]
end

# Normalize the data
volume_min, volume_max = minimum(interpolation), maximum(interpolation)
time_min, time_max = minimum(t_values), maximum(t_values)

interpolation_norm = (interpolation .- volume_min) ./ (volume_max - volume_min)
t_values_norm = (collect(t_values) .- time_min) ./ (time_max - time_min) 
u0_norm = [interpolation_norm[1]]
tspan_norm = (0.0, 1.0)

# Create ODE problem with normalized data
prob_neuralode = ODEProblem(neural_ode_func!, u0_norm, tspan_norm, p)

# Prediction function
function predict_neuralode(p)
    prob = ODEProblem(neural_ode_func!, u0_norm, tspan_norm, p)
    sol = solve(prob, Tsit5(), saveat=t_values_norm, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    return Array(sol)[1, :]
end

# Loss function
function loss_neuralode(p)
    pred = predict_neuralode(p)
    
    data_loss = sum(abs2, interpolation_norm .- pred)
    
    # Derivative matching loss
    if length(pred) > 1
        pred_derivs = diff(pred)
        true_derivs = diff(interpolation_norm)
        deriv_loss = sum(abs2, true_derivs .- pred_derivs)
        
        # Smoothness
        if length(pred_derivs) > 1
            second_derivs = diff(pred_derivs)
            smooth_loss = 0.01 * sum(abs2, second_derivs)
        else
            smooth_loss = 0.0
        end
        
        total_loss = data_loss + 0.1 * deriv_loss + smooth_loss
    else
        total_loss = data_loss
    end
    
    return total_loss, pred
end

# Callback function
callback = function (p, l, pred; doplot = true)
    println(l)
    if doplot && length(pred) == length(interpolation_norm)
        # Denormalize for plotting
        pred_denorm = pred .* (volume_max - volume_min) .+ volume_min
        
        plt = scatter(t_values, interpolation, label="Training Data", color=:mediumpurple, markersize = 3)
        plot!(plt, t_values, pred_denorm, label="Neural ODE", color=:red, lw=2)
        xlabel!("Time (days)")
        ylabel!("Tumor Volume (mm³)")
        title!("Neural ODE Prediction Vs Training Data")
        display(plt)
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot = true)

# Optimization
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_final = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback=callback, maxiters=500)

# Denormalize
final_pred = predict_neuralode(result_final.u)
final_pred_denorm = final_pred .* (volume_max - volume_min) .+ volume_min

# Plot the Neural ODE solution
plt = scatter(t_values, interpolation, label="Training Data", color=:mediumpurple, markersize = 3)
plot!(plt, t_values, final_pred_denorm, label="Neural ODE Prediction", color=:red, lw=3)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
display(plt)