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


# --- DATA SPLITTING (90-10) ---

# Calculate split point (90% of data for training)
n_total = length(t_values)
n_train = Int(floor(0.9 * n_total)) # Change to 0.8 or 0.7 for 80% or 70% training

# Split the data
t_train = t_values[1:n_train]
t_test = t_values[n_train:end]
volume_train = interpolation[1:n_train]
volume_test = interpolation[n_train:end]


# --- GOMPERTZ ODE (TRAINING ON 90% OF DATA) ---

# Define the ODE function
function gompertz!(du, u, p, t)
    V = u[1]
    (a, K) = p
    du[1] = a * V * log(K / V)
end

# Define the ODE problem for training data
u0 = [volume_train[1]]
p0 = [0.02, 55000.0]  # parameters [a, K] - change based on the subject; a = 0.3 for mouse, a = 0.02 for human
tspan_train = (minimum(t_train), maximum(t_train))

prob_train = ODEProblem(gompertz!, u0, tspan_train, p0)
sol_train = solve(prob_train, TRBDF2(), saveat=t_train)

# Training prediction
train_pred = [u[1] for u in sol_train.u]

# Plot training fit
plt = scatter(t_train, volume_train, label="Training Data", color=:mediumpurple, markersize=4)
plot!(plt, t_train, train_pred, label="Gompertz Fit", color=:red, lw=3)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
title!("Gompertz ODE Training (90% of data)")
display(plt)

# --- FORECASTING ON REMAINING 10% ---

# Continue from end of training
u0_forecast = [train_pred[end]]
tspan_forecast = (maximum(t_train), maximum(t_test))

prob_forecast = ODEProblem(gompertz!, u0_forecast, tspan_forecast, p0)
sol_forecast = solve(prob_forecast, TRBDF2(), saveat=t_test)

# Forecast values
forecast_values = [u[1] for u in sol_forecast.u]

# Plot the complete picture: training + forecast vs actual
plt = scatter(t_train, volume_train, label="Training Data (90%)", color=:mediumpurple, markersize=3)
scatter!(plt, t_test, volume_test, label="Test Data (10%)", color=:orange, markersize=3)
plot!(plt, t_train, train_pred, label="Gompertz Training", color=:red, lw=2)
plot!(plt, t_test, forecast_values, label="Gompertz Forecast", color=:green, lw=2, linestyle=:dash)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
vline!([maximum(t_train)], color=:black, linestyle=:dot, label="Train/Test Split")
display(plt)


# Normalize the training data
volume_min, volume_max = minimum(volume_train), maximum(volume_train)
time_min, time_max = minimum(t_train), maximum(t_train)

volume_train_norm = (volume_train .- volume_min) ./ (volume_max - volume_min)
t_train_norm = (collect(t_train) .- time_min) ./ (time_max - time_min) 
u0_norm = [volume_train_norm[1]]
tspan_norm = (0.0, 1.0)


# --- GRU (TRAINING ON 90% OF DATA) ---

# Create sequences for GRU training
# seq_length: number of past timesteps to use as input
function create_sequences(times, volumes; seq_length=5)
    n = length(volumes)
    X = []  # Input sequences
    Y = []  # Target values
    
    for i in seq_length:n-1
        # Input: past seq_length (time, volume) pairs
        seq = vcat(
            reshape(times[i-seq_length+1:i], 1, seq_length),
            reshape(volumes[i-seq_length+1:i], 1, seq_length)
        )  # Shape: (2, seq_length)
        push!(X, seq)
        
        # Target: next volume value
        push!(Y, volumes[i+1])
    end
    
    return X, Y
end


# Create training sequences
seq_length = 8 # can tune this
X_train, Y_train = create_sequences(t_train_norm, volume_train_norm; seq_length=seq_length)

# Convert to appropriate format for Flux
# X: list of matrices (2, seq_length)
# Y: vector of targets
Y_train = Float32.(Y_train)

# Define GRU Model
# Model architecture
input_size = 2       # (time, volume)
hidden_size = 16     # hidden units
output_size = 1      # predict next volume

model = Flux.Chain(
    Flux.GRU(input_size => hidden_size),
    Flux.Dense(hidden_size => output_size, sigmoid) 
)

function predict_sequence(model, seq)
    Flux.reset!(model)
    
    # seq is (2, seq_length) - process each timestep
    local output
    for t in 1:size(seq, 2)
        x_t = seq[:, t:t]  # (2, 1) column vector
        output = model(x_t)
    end
    return output[1]  # scalar prediction
end

function loss_fn(model, X, Y)
    total_loss = 0.0f0
    for (seq, target) in zip(X, Y)
        pred = predict_sequence(model, Float32.(seq))
        total_loss += (pred - target)^2
    end
    return total_loss / length(Y)
end

# Optimizer
opt_state = Flux.setup(Flux.Adam(0.01), model)

epochs = 500
losses = Float32[]

println("Starting GRU Training...")

for epoch in 1:epochs
    # Compute gradients
    grads = Flux.gradient(model) do m
        loss_fn(m, X_train, Y_train)
    end
    
    # Update parameters
    Flux.update!(opt_state, model, grads[1])
    
    # Track loss
    current_loss = loss_fn(model, X_train, Y_train)
    push!(losses, current_loss)
    
    if epoch % 50 == 0 || epoch == 1
        println("Epoch $epoch / $epochs | Loss: $(round(current_loss, digits=6))")
    end
end

println("Final Loss: $(round(losses[end], digits=6))")

# Prediction function
function predict_all(model, X)
    predictions = Float32[]
    for seq in X
        pred = predict_sequence(model, Float32.(seq))
        push!(predictions, pred)
    end
    return predictions
end

# Get final training prediction
final_pred_train = predict_all(model, X_train)
final_pred_train_denorm = final_pred_train .* (volume_max - volume_min) .+ volume_min
t_pred_train = collect(t_train)[seq_length+1:end]

plt = scatter(collect(t_train), volume_train, label="Training Data", color=:mediumpurple, markersize=4)
plot!(plt, t_pred_train, final_pred_train_denorm, label="GRU Final", color=:red, lw=3)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
title!("Final GRU Prediction")
display(plt)

# --- FORECASTING ON REMAINING 10% ---

# Normalize test times using training statistics
t_test_norm = (collect(t_test) .- time_min) ./ (time_max - time_min)

# Forecasting function
function forecast_gru(model, times_train_norm, volumes_train_norm, times_test_norm; seq_length=8)
    # Start with the last seq_length points from training
    current_times = collect(Float32, times_train_norm[end-seq_length+1:end])
    current_volumes = collect(Float32, volumes_train_norm[end-seq_length+1:end])
    
    forecasts = Float32[]
    
    for t_new in times_test_norm
        # Create input sequence
        seq = vcat(
            reshape(current_times, 1, seq_length),
            reshape(current_volumes, 1, seq_length)
        )
        
        # Predict next value
        Flux.reset!(model)
        pred = predict_sequence(model, seq)
        push!(forecasts, pred)
        
        # Update sliding window (shift and add new prediction)
        current_times = vcat(current_times[2:end], Float32(t_new))
        current_volumes = vcat(current_volumes[2:end], pred)
    end
    
    return forecasts
end

# Generate forecasts
forecast_values_norm = forecast_gru(model, t_train_norm, volume_train_norm, t_test_norm; seq_length=seq_length)

# Denormalize forecast values
forecast_values = forecast_values_norm .* (volume_max - volume_min) .+ volume_min

# Plot the complete picture: training + forecast vs actual
plt = scatter(collect(t_train), volume_train, label="Training Data (90%)", color=:mediumpurple, markersize=3)
scatter!(plt, collect(t_test), volume_test, label="Test Data (10%)", color=:orange, markersize=3)
plot!(plt, t_pred_train, final_pred_train_denorm, label="GRU Training", color=:red, lw=2)
plot!(plt, collect(t_test), forecast_values, label="GRU Forecast", color=:green, lw=2, linestyle=:dash)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
vline!([maximum(t_train)], color=:black, linestyle=:dot, label="Train/Test Split")
display(plt)


# --- LSTM (TRAINING ON 90% OF DATA) ---

# Create sequences for LSTM training
function create_sequences(times, volumes; seq_length=5)
    n = length(volumes)
    X = []  # Input sequences
    Y = []  # Target values
    
    for i in seq_length:n-1
        # Input: past seq_length (time, volume) pairs
        seq = vcat(
            reshape(times[i-seq_length+1:i], 1, seq_length),
            reshape(volumes[i-seq_length+1:i], 1, seq_length)
        )  # Shape: (2, seq_length)
        push!(X, seq)
        
        # Target: next volume value
        push!(Y, volumes[i+1])
    end
    
    return X, Y
end

# Create training sequences
seq_length = 10
X_train, Y_train = create_sequences(t_train_norm, volume_train_norm; seq_length=seq_length)

# Convert to float32
Y_train = Float32.(Y_train)


# Model architecture
input_size = 2       # (time, volume) at each timestep
hidden_size = 16     # LSTM hidden units
output_size = 1      # predict next volume

# LSTM model
model = Flux.Chain(
    Flux.LSTM(input_size => hidden_size),
    Flux.Dense(hidden_size => output_size, sigmoid)
)

function predict_sequence(model, seq)
    Flux.reset!(model)
    
    # seq is (2, seq_length) - process each timestep
    local output
    for t in 1:size(seq, 2)
        x_t = seq[:, t:t]  # (2, 1) column vector
        output = model(x_t)
    end
    return output[1]  # scalar prediction
end


function loss_fn(model, X, Y)
    total_loss = 0.0f0
    for (seq, target) in zip(X, Y)
        pred = predict_sequence(model, Float32.(seq))
        total_loss += (pred - target)^2
    end
    return total_loss / length(Y)
end

# Optimizer
opt_state = Flux.setup(Flux.Adam(0.01), model)

epochs = 300
losses = Float32[]

println("Starting LSTM Training...")

for epoch in 1:epochs
    # Compute gradients
    grads = Flux.gradient(model) do m
        loss_fn(m, X_train, Y_train)
    end
    
    # Update parameters
    Flux.update!(opt_state, model, grads[1])
    
    # Track loss
    current_loss = loss_fn(model, X_train, Y_train)
    push!(losses, current_loss)
    
    if epoch % 50 == 0 || epoch == 1
        println("Epoch $epoch / $epochs | Loss: $(round(current_loss, digits=6))")
    end
end


println("Final Loss: $(round(losses[end], digits=6))")

function predict_all(model, X)
    predictions = Float32[]
    for seq in X
        pred = predict_sequence(model, Float32.(seq))
        push!(predictions, pred)
    end
    return predictions
end


# Get final training prediction
final_pred_train = predict_all(model, X_train)
final_pred_train_denorm = final_pred_train .* (volume_max - volume_min) .+ volume_min
t_pred_train = collect(t_train)[seq_length+1:end]

plt = scatter(collect(t_train), volume_train, label="Training Data", color=:mediumpurple, markersize=4)
plot!(plt, t_pred_train, final_pred_train_denorm, label="LSTM Final", color=:red, lw=3)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
title!("Final LSTM Prediction")
display(plt)

# --- FORECASTING ON REMAINING 10% ---

# Normalize test times using training statistics
t_test_norm = (collect(t_test) .- time_min) ./ (time_max - time_min)

# Forecasting function
function forecast_lstm(model, times_train_norm, volumes_train_norm, times_test_norm; seq_length=10)
    # Start with the last seq_length points from training
    current_times = collect(Float32, times_train_norm[end-seq_length+1:end])
    current_volumes = collect(Float32, volumes_train_norm[end-seq_length+1:end])
    
    forecasts = Float32[]
    
    for t_new in times_test_norm
        # Create input sequence
        seq = vcat(
            reshape(current_times, 1, seq_length),
            reshape(current_volumes, 1, seq_length)
        )
        
        # Predict next value
        Flux.reset!(model)
        pred = predict_sequence(model, seq)
        push!(forecasts, pred)
        
        # Update sliding window (shift and add new prediction)
        current_times = vcat(current_times[2:end], Float32(t_new))
        current_volumes = vcat(current_volumes[2:end], pred)
    end
    
    return forecasts
end

# Generate forecasts
forecast_values_norm = forecast_lstm(model, t_train_norm, volume_train_norm, t_test_norm; seq_length=seq_length)

# Denormalize forecast values
forecast_values = forecast_values_norm .* (volume_max - volume_min) .+ volume_min

# Plot the complete picture: training + forecast vs actual
plt = scatter(collect(t_train), volume_train, label="Training Data (90%)", color=:mediumpurple, markersize=3)
scatter!(plt, collect(t_test), volume_test, label="Test Data (10%)", color=:orange, markersize=3)
plot!(plt, t_pred_train, final_pred_train_denorm, label="LSTM Training", color=:red, lw=2)
plot!(plt, collect(t_test), forecast_values, label="LSTM Forecast", color=:green, lw=2, linestyle=:dash)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
vline!([maximum(t_train)], color=:black, linestyle=:dot, label="Train/Test Split")
display(plt)


# --- NEURAL ODE (TRAINING ON 90% OF DATA) ---

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

# Create ODE problem with normalized training data
prob_neuralode = ODEProblem(neural_ode_func!, u0_norm, tspan_norm, p)

# Prediction function
function predict_neuralode_train(p)
    prob = ODEProblem(neural_ode_func!, u0_norm, tspan_norm, p)
    sol = solve(prob, Tsit5(), saveat=t_train_norm, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    return Array(sol)[1, :]
end

# Loss function
function loss_neuralode(p)
    pred = predict_neuralode_train(p)
    data_loss = sum(abs2, volume_train_norm .- pred)
    if length(pred) > 1
        pred_derivs = diff(pred)
        true_derivs = diff(volume_train_norm)
        deriv_loss = sum(abs2, true_derivs .- pred_derivs)
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
    if doplot && length(pred) == length(volume_train_norm)
        # Denormalize for plotting
        pred_denorm = pred .* (volume_max - volume_min) .+ volume_min
        
        plt = scatter(t_train, volume_train, label="Training Data", color=:mediumpurple, markersize=3)
        plot!(plt, t_train, pred_denorm, label="Neural ODE", color=:red, lw=2)
        xlabel!("Time (days)")
        ylabel!("Tumor Volume (mm³)")
        title!("Neural ODE Training (90% of data)")
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

result_final = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.005), callback=callback, maxiters=500)

# Get final training prediction
final_pred_train = predict_neuralode_train(result_final.u)
final_pred_train_denorm = final_pred_train .* (volume_max - volume_min) .+ volume_min

plt = scatter(t_train, volume_train, label="Training Data", color=:mediumpurple, markersize=4)
plot!(plt, t_train, final_pred_train_denorm, label="Neural ODE Final", color=:red, lw=3)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
title!("Final Neural ODE Prediction")
display(plt)

# --- FORECASTING ON REMAINING 10% ---

optimized_params = result_final.u

# Neural ODE function for forecasting 
function neural_ode_forecast_func!(du, u, p, t)
    t_norm = (t - time_min) / (time_max - time_min)
    inputs = vcat(u, [t_norm])
    y_pred, _ = dudt_nn(inputs, p, st)
    du[1] = y_pred[1]
end

# Set up forecasting
tspan_forecast = (maximum(t_train), maximum(t_test))
saveat_forecast = collect(t_test)

# Use the final state from training as initial condition for forecast
u0_forecast_norm = [final_pred_train[end]]

# Create forecast problem 
prob_forecast = ODEProblem(neural_ode_forecast_func!, u0_forecast_norm, tspan_forecast, optimized_params)

# Solve the forecast
sol_forecast = solve(prob_forecast, Tsit5(), saveat=saveat_forecast)

# Denormalize forecast values
forecast_values_norm = [u[1] for u in sol_forecast.u]
forecast_values = forecast_values_norm .* (volume_max - volume_min) .+ volume_min

# Plot the complete picture: training + forecast vs actual
plt = scatter(t_train, volume_train, label="Training Data (90%)", color=:mediumpurple, markersize=3)
scatter!(plt, t_test, volume_test, label="Test Data (10%)", color=:orange, markersize=3)
plot!(plt, t_train, final_pred_train_denorm, label="Neural ODE Training", color=:red, lw=2)
plot!(plt, t_test, forecast_values, label="Neural ODE Forecast", color=:green, lw=2, linestyle=:dash)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
vline!([maximum(t_train)], color=:black, linestyle=:dot, label="Train/Test Split")
display(plt)

# --- UDE (TRAINING ON 90% OF DATA) ---

rng = Xoshiro(123)

# Normalization
V_mean = mean(volume_train)
V_std = std(volume_train)

t_min = minimum(t_train)
t_max = maximum(t_train)

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

u0_raw = [volume_train[1]]
tspan_raw = (minimum(t_train), maximum(t_train))

# Prediction function
function predict_ude_train(p)
    prob = ODEProblem(ude_func!, u0_raw, tspan_raw, p)
    sol = solve(prob, Tsit5(), saveat=t_train)
    return Array(sol)[1, :]
end

# Loss function
function loss_ude(p)
    pred = predict_ude_train(p)
    data_loss = mean(abs2, volume_train .- pred)
    if length(pred) > 1
        pred_derivs = diff(pred)
        true_derivs = diff(volume_train)
        deriv_loss = mean(abs2, true_derivs .- pred_derivs)
        smooth_loss = length(pred_derivs) > 1 ? 0.01 * mean(abs2, diff(pred_derivs)) : 0.0
        total_loss = data_loss + 0.5 * deriv_loss + 0.05 * smooth_loss
    else
        total_loss = data_loss
    end
    return total_loss, pred
end

# Callback function
callback = function (p, l, pred; doplot = true)
    println(l)
    if doplot && length(pred) == length(volume_train)
        plt = scatter(t_train, volume_train, label="Training Data", color=:mediumpurple, markersize = 3)
        plot!(plt, t_train, pred, label="UDE", color=:red, lw=2)
        xlabel!("Time (days)")
        ylabel!("Tumor Volume (mm³)")
        title!("UDE Training (90% of data)")
        display(plt)
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

# Get final training prediction
final_pred_train = predict_ude_train(result_final.u)

plt = scatter(t_train, volume_train, label="Training Data", color=:mediumpurple, markersize=4)
plot!(plt, t_train, final_pred_train, label="UDE Final", color=:red, lw=3)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
title!("Final UDE Prediction")
display(plt)

# --- FORECASTING ON REMAINING 10% ---

optimized_params = result_final.u

# UDE function for forecasting 
function ude_forecast_func!(du, u, p, t)
    V = u[1]
    V_norm = (V - V_mean) / V_std
    t_norm = (t - t_min) / (t_max - t_min)
    inputs = reshape([V_norm, t_norm], :, 1)
    NN_a_output, _ = NN1(inputs, p.layer_1, st1)
    NN_log_output, _ = NN2(inputs, p.layer_2, st2)
    NN_a = log1p(exp(NN_a_output[1]))
    NN_log = NN_log_output[1]
    du[1] = NN_a * V * NN_log
end

# Set up forecasting
tspan_forecast = (t_train[end], t_test[end])
u0_forecast = [final_pred_train[end]]
saveat_forecast = t_test

# Create forecast problem 
prob_forecast = ODEProblem(ude_forecast_func!, u0_forecast, tspan_forecast, optimized_params)

# Solve the forecast
sol_forecast = solve(prob_forecast, Tsit5(), saveat=saveat_forecast)

forecast_values = [u[1] for u in sol_forecast.u]

# Plot the complete picture: training + forecast vs actual
plt = scatter(t_train, volume_train, label="Training Data (90%)", color=:mediumpurple, markersize=3)
scatter!(plt, t_test, volume_test, label="Test Data (10%)", color=:orange, markersize=3)
plot!(plt, t_train, final_pred_train, label="UDE Training", color=:red, lw=2)
plot!(plt, t_test, forecast_values, label="UDE Forecast", color=:green, lw=2, linestyle=:dash)
xlabel!("Time (days)")
ylabel!("Tumor Volume (mm³)")
vline!([t_train[end]], color=:black, linestyle=:dot, label="Train/Test Split")
display(plt)