import jax
import jax.numpy as jnp
from config import alpha, x_max, x_min, t_min, t_max
import matplotlib.pyplot as plt
from physics import u_model_vmap

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

def sample_training_points(key, N_f, N_bc, N_ic):
    """
    Sample training points for:
    1. PDE residual inside the domain
    2. Boundary conditions on x=0 and x=1
    3. Initial condition on t=0

    Args: 
        - N_f, N_bc, N_ic = collocation points described in problem 
        - key = random key to be split up
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

def plot_results(params):
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

def plot_training_loss(loss_history):
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