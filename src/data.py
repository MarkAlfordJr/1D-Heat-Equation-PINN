import jax
import jax.numpy as jnp

def sample_training_points(key, config):
    """
    Makes interior collocation, boundary condition collocation, and initial conditions
    collocation points from the problem settings/hyperparameters.
    """
    x_min = config["problem"]["x_min"]
    x_max = config["problem"]["x_max"]
    t_min = config["problem"]["t_min"]
    t_max = config["problem"]["t_max"]

    N_f = config["data"]["N_f"]
    N_bc = config["data"]["N_bc"]
    N_ic = config["data"]["N_ic"]

    k1, k2, k3, k4 = jax.random.split(key, 4)

    x_f = jax.random.uniform(k1, (N_f,), minval=x_min, maxval=x_max)
    t_f = jax.random.uniform(k2, (N_f,), minval=t_min, maxval=t_max)

    t_bc = jax.random.uniform(k3, (N_bc,), minval=t_min, maxval=t_max)
    x_bc_left = jnp.zeros((N_bc,))
    x_bc_right = jnp.ones((N_bc,))

    x_ic = jax.random.uniform(k4, (N_ic,), minval=x_min, maxval=x_max)
    t_ic = jnp.zeros((N_ic,))

    return {
        "x_f": x_f,
        "t_f": t_f,
        "x_bc_left": x_bc_left,
        "x_bc_right": x_bc_right,
        "t_bc": t_bc,
        "x_ic": x_ic,
        "t_ic": t_ic,
    }