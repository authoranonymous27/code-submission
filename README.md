# Anonymous Code Submission

## Running the code
To reproduce the environment, clone the repository and navigate to the folder:
```
git clone https://github.com/authoranonymous27/practice-code-submission.git
cd practice-code-submission
```
Then, instantiate the environment:
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
This installs the exact package versions specified in Project.toml and Manifest.toml. 
You can now run the scripts directly from the terminal:
```
julia --project=. <filename>.jl
```
Or from inside the Julia REPL:
```
julia --project=.
julia> include("<filename>.jl")
```
For instructions on installing Julia, see: https://julialang.org/install/

## Repository Structure
```
.
├── UDE.jl
├── NODE.jl
├── Gompertz.jl
├── GRU.jl
├── LSTM.jl
├── SymbolicRecovery.jl
├── Forecasting.jl
├── data/
│   ├── mouse_data.txt
│   └── preprocessed_tumor_data.csv
├── plots/
│   ├── human/
│     ├── 90-10/ ...
│     ├── 80-20/ ...
│     └── 70-30/ ...
│   └── mouse/
│     ├── 90-10/ ...
│     ├── 80-20/ ...
│     └── 70-30/ ...
├── Manifest.toml
├── Project.toml
└── README.md
```

Dataset Sources:
