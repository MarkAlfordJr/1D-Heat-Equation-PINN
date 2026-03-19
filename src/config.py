"""
    THE CONFIG FILE
    - stores the hyperparameters and problem settings
"""
# spatial domain for u(x,t)
# Define spatial domain: x goes from 0 to 1
x_min, x_max = 0.0, 0.1

# Define time domain: t goes from 0 to 1
t_min, t_max = 0.0, 1.0

# Diffusion coefficient in the heat equation
alpha = 0.1

# Number of interior collocation points used to enforce the PDE residual
N_f = 1024

# Number of boundary points used to enforce boundary conditions
N_bc = 256

# Number of initial-condition points used to enforce u(x,0)
N_ic = 256

# Optimizer learning rate
learning_rate = 1e-3

# Number of training iterations
epochs = 5000

# How often to print training progress
print_every = 500

# NN layer architecture
LAYER_SIZES = [2,64,64,64,1]