# Import JAX itself
import jax

# Import JAX's NumPy-like API
# This behaves like NumPy, but works with JAX transformations like grad, vmap, and jit
import jax.numpy as jnp

# Optax is a gradient optimization library built for JAX
# We use it for the Adam optimizer
import optax

# Matplotlib is only for plotting results after training
import matplotlib.pyplot as plt

# partial is useful in many JAX projects, though it is not strictly necessary here
from functools import partial


# ============================================================
# 1. Problem setup
# ============================================================

# Diffusion coefficient in the heat equation
# The PDE is u_t = alpha * u_xx
alpha = 0.1

# Define spatial domain: x goes from 0 to 1
x_min, x_max = 0.0, 1.0

# Define time domain: t goes from 0 to 1
t_min, t_max = 0.0, 1.0

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

# Neural network architecture:
# input dimension = 2 because inputs are (x, t)
# output dimension = 1 because output is scalar u(x,t)
layer_sizes = [2, 64, 64, 64, 1]


# ============================================================
# 2. Simple MLP from scratch
# ============================================================

def init_mlp(layer_sizes, key):
    """
    Initialize MLP parameters.

    Returns:
        params = [(W1, b1), (W2, b2), ...]
    where each W is a weight matrix and each b is a bias vector.
    """
    params = []

    # Create one random key per layer
    keys = jax.random.split(key, len(layer_sizes) - 1)

    # Loop through each pair of consecutive layer sizes
    # Example: (2 -> 64), (64 -> 64), (64 -> 64), (64 -> 1)
    for k, (in_dim, out_dim) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        # Xavier-style initialization limit
        # This helps avoid weights being too large or too small initially
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))

        # Initialize weights uniformly in [-limit, limit]
        W = jax.random.uniform(k, (in_dim, out_dim), minval=-limit, maxval=limit)

        # Initialize biases to zero
        b = jnp.zeros((out_dim,))

        # Store this layer's parameters
        params.append((W, b))

    return params


def mlp_forward(params, x):
    """
    Forward pass through the neural network for ONE input vector x.

    Args:
        params: list of (W, b) pairs
        x: input vector of shape (2,) here, since input is [x, t]

    Returns:
        scalar output u(x,t)
    """
    # 'a' stands for the current activation vector flowing through the network
    a = x

    # Iterate through each layer
    for i, (W, b) in enumerate(params):
        # Compute affine transformation z = aW + b
        z = jnp.dot(a, W) + b

        # For all hidden layers, apply tanh activation
        if i < len(params) - 1:
            a = jnp.tanh(z)
        else:
            # For the final layer, use no activation
            # because we want a raw scalar output
            a = z

    # Final output has shape (1,), so take the first element
    return a[0]


# ============================================================
# 3. PINN model u_theta(x, t)
# ============================================================

def u_model(params, x, t):
    """
    Define the PINN approximation u_theta(x,t).

    The network takes x and t as inputs and predicts a scalar u.
    """
    # Combine x and t into one input vector [x, t]
    inp = jnp.array([x, t])

    # Feed [x, t] through the MLP
    return mlp_forward(params, inp)


# ============================================================
# 4. Autodiff: derivatives needed for PDE residual
# ============================================================

def u_t(params, x, t):
    """
    Compute partial derivative of u with respect to t.

    jax.grad differentiates a scalar-output function with respect
    to its argument. Here the variable is tt.
    """
    return jax.grad(lambda tt: u_model(params, x, tt))(t)


def u_x(params, x, t):
    """
    Compute partial derivative of u with respect to x.
    """
    return jax.grad(lambda xx: u_model(params, xx, t))(x)


def u_xx(params, x, t):
    """
    Compute second partial derivative with respect to x.

    We do this by differentiating u_x with respect to x again.
    """
    return jax.grad(lambda xx: u_x(params, xx, t))(x)


def pde_residual(params, x, t):
    """
    Heat equation residual:
        f(x,t) = u_t - alpha * u_xx

    If the PDE is satisfied exactly, this should be zero.
    """
    return u_t(params, x, t) - alpha * u_xx(params, x, t)


# Vectorize u_model over batches of x and t values
# in_axes=(None, 0, 0) means:
# - params is shared (not batched)
# - x is batched along axis 0
# - t is batched along axis 0
u_model_vmap = jax.vmap(u_model, in_axes=(None, 0, 0))

# Vectorize PDE residual the same way
pde_residual_vmap = jax.vmap(pde_residual, in_axes=(None, 0, 0))


# ============================================================
# 5. Exact solution (for evaluation only)
# ============================================================

def exact_solution(x, t):
    """
    Known analytical solution for:
        u_t = alpha u_xx
        u(0,t)=0
        u(1,t)=0
        u(x,0)=sin(pi x)

    This is NOT used in training.
    It is only used after training to check how close the PINN is.
    """
    return jnp.exp(-(jnp.pi ** 2) * alpha * t) * jnp.sin(jnp.pi * x)


# ============================================================
# 6. Sampling points
# ============================================================

def sample_training_points(key, N_f, N_bc, N_ic):
    """
    Sample training points for:
    1. PDE residual inside the domain
    2. Boundary conditions on x=0 and x=1
    3. Initial condition on t=0
    """

    # Split one random key into four smaller keys
    # so each sampling step uses independent randomness
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # -----------------------------
    # Interior PDE collocation points
    # -----------------------------

    # Random x values inside [0,1]
    x_f = jax.random.uniform(k1, (N_f,), minval=x_min, maxval=x_max)

    # Random t values inside [0,1]
    t_f = jax.random.uniform(k2, (N_f,), minval=t_min, maxval=t_max)

    # -----------------------------
    # Boundary condition points
    # -----------------------------

    # Random times for boundary enforcement
    t_bc = jax.random.uniform(k3, (N_bc,), minval=t_min, maxval=t_max)

    # Left boundary x=0
    x_bc_left = jnp.zeros((N_bc,))

    # Right boundary x=1
    x_bc_right = jnp.ones((N_bc,))

    # -----------------------------
    # Initial condition points
    # -----------------------------

    # Random x positions along the initial line t=0
    x_ic = jax.random.uniform(k4, (N_ic,), minval=x_min, maxval=x_max)

    # Initial condition occurs at t=0
    t_ic = jnp.zeros((N_ic,))

    return x_f, t_f, x_bc_left, x_bc_right, t_bc, x_ic, t_ic


# ============================================================
# 7. Loss function
# ============================================================

def loss_fn(params, batch):
    """
    Compute total PINN loss.

    batch contains:
      x_f, t_f         -> interior PDE points
      x_bc_left/right  -> boundary x values
      t_bc             -> boundary times
      x_ic, t_ic       -> initial-condition points
    """
    x_f, t_f, x_bc_left, x_bc_right, t_bc, x_ic, t_ic = batch

    # -----------------------------
    # PDE residual loss
    # -----------------------------

    # Evaluate PDE residual at all interior collocation points
    f_vals = pde_residual_vmap(params, x_f, t_f)

    # Mean squared PDE residual
    # If residual is small, network approximately satisfies PDE
    loss_pde = jnp.mean(f_vals ** 2)

    # -----------------------------
    # Boundary condition loss
    # -----------------------------

    # Predict u at left boundary x=0
    u_left = u_model_vmap(params, x_bc_left, t_bc)

    # Predict u at right boundary x=1
    u_right = u_model_vmap(params, x_bc_right, t_bc)

    # Since boundary condition is u=0 at both ends,
    # we penalize squared deviation from zero
    loss_bc = jnp.mean(u_left ** 2) + jnp.mean(u_right ** 2)

    # -----------------------------
    # Initial condition loss
    # -----------------------------

    # Network prediction at t=0
    u_ic_pred = u_model_vmap(params, x_ic, t_ic)

    # True initial condition is sin(pi x)
    u_ic_true = jnp.sin(jnp.pi * x_ic)

    # Penalize mismatch with initial condition
    loss_ic = jnp.mean((u_ic_pred - u_ic_true) ** 2)

    # -----------------------------
    # Total loss
    # -----------------------------

    # Standard PINN total loss = PDE + boundary + initial
    total = loss_pde + loss_bc + loss_ic

    # Return total plus the individual parts for monitoring
    return total, (loss_pde, loss_bc, loss_ic)


# ============================================================
# 8. Optimizer
# ============================================================

# Create Adam optimizer
optimizer = optax.adam(learning_rate)


@jax.jit
def train_step(params, opt_state, batch):
    """
    One training step:
    1. Compute loss
    2. Compute gradients of loss wrt params
    3. Update parameters with Adam
    """

    # value_and_grad computes both:
    # - the loss value
    # - the gradient of that loss wrt params
    #
    # has_aux=True means loss_fn returns:
    #   (main_loss, extra_info)
    # where extra_info here is (loss_pde, loss_bc, loss_ic)
    (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)

    # Use optimizer to turn gradients into parameter updates
    updates, opt_state = optimizer.update(grads, opt_state, params)

    # Apply updates to parameters
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss_value, aux


# ============================================================
# 9. Training loop
# ============================================================

# Create a master random key
key = jax.random.PRNGKey(0)

# Split it so one part initializes the network
key, init_key = jax.random.split(key)

# Initialize neural network parameters
params = init_mlp(layer_sizes, init_key)

# Initialize optimizer state for Adam
opt_state = optimizer.init(params)

# Store loss values for plotting later
loss_history = []

# Main training loop
for epoch in range(1, epochs + 1):
    # Split random key each epoch so we can resample fresh points
    key, batch_key = jax.random.split(key)

    # Sample a new batch of collocation, boundary, and initial points
    batch = sample_training_points(batch_key, N_f, N_bc, N_ic)

    # Perform one optimization step
    params, opt_state, loss_value, aux = train_step(params, opt_state, batch)

    # Unpack the separate loss terms
    loss_pde, loss_bc, loss_ic = aux

    # Save total loss for plotting
    loss_history.append(float(loss_value))

    # Print progress every print_every epochs
    if epoch % print_every == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Total: {loss_value:.6e} | "
            f"PDE: {loss_pde:.6e} | "
            f"BC: {loss_bc:.6e} | "
            f"IC: {loss_ic:.6e}"
        )


# ============================================================
# 10. Evaluate on a grid
# ============================================================

# Create 200 x-values for plotting
x_plot = jnp.linspace(0.0, 1.0, 200)

# Choose several time slices to inspect
times_to_plot = [0.0, 0.25, 0.5, 0.75, 1.0]

# Create a new figure
plt.figure(figsize=(8, 5))

# Loop over selected times
for t_val in times_to_plot:
    # Create an array full of the same time value
    # so we can evaluate u(x, t_val) across all x_plot
    t_array = jnp.full_like(x_plot, t_val)

    # PINN prediction
    u_pred = u_model_vmap(params, x_plot, t_array)

    # Exact solution for comparison
    u_true = exact_solution(x_plot, t_array)

    # Plot PINN prediction
    plt.plot(x_plot, u_pred, label=f"PINN t={t_val:.2f}")

    # Plot exact solution as dashed line
    plt.plot(x_plot, u_true, "--", label=f"Exact t={t_val:.2f}")

# Label axes
plt.xlabel("x")
plt.ylabel("u(x,t)")

# Title
plt.title("1D Heat Equation solved by a PINN in JAX")

# Show legend
plt.legend(fontsize=8, ncol=2)

# Adjust spacing
plt.tight_layout()

# Display plot
plt.show()


# ============================================================
# 11. Plot training loss
# ============================================================

# New figure for training loss
plt.figure(figsize=(7, 4))

# Plot total loss over epochs
plt.plot(loss_history)

# Use logarithmic y-scale because PINN losses often span many orders of magnitude
plt.yscale("log")

# Axis labels
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Title
plt.title("Training Loss")

# Adjust layout
plt.tight_layout()

# Display plot
plt.show()