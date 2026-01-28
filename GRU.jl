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


# --- GRU ---

# Set random seed for reproducibility
Random.seed!(123)

# Normalization
volume_min, volume_max = minimum(interpolation), maximum(interpolation)
time_min, time_max = minimum(t_values), maximum(t_values)

# Normalize data to [0, 1]
volumes_norm = (interpolation .- volume_min) ./ (volume_max - volume_min)
times_norm = (t_values .- time_min) ./ (time_max - time_min)


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
seq_length = 8
X_train, Y_train = create_sequences(times_norm, volumes_norm; seq_length=seq_length)

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
    local output
    for t in 1:size(seq, 2)
        x_t = seq[:, t:t] 
        output = model(x_t)
    end
    return output[1] 
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

# Generate predictions
function predict_all(model, X)
    predictions = Float32[]
    for seq in X
        pred = predict_sequence(model, Float32.(seq))
        push!(predictions, pred)
    end
    return predictions
end

predictions_norm = predict_all(model, X_train)

# Denormalize predictions
predictions = predictions_norm .* (volume_max - volume_min) .+ volume_min

# Corresponding actual values (denormalized)
actuals = Y_train .* (volume_max - volume_min) .+ volume_min

# Time points for predictions (offset by seq_length)
t_pred = t_values[seq_length+1:end]



# Plot GRU solution
p = scatter(t_values, interpolation, label="Interpolated Data", color=:mediumpurple, markersize=3, alpha=0.7)
plot!(p, t_pred, predictions, label="GRU Predictions", color=:red, lw=2)
xlabel!(p, "Time (days)")
ylabel!(p, "Tumor Volume (mm³)")