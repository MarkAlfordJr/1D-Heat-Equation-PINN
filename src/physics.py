import jax
import jax.numpy as jnp
from src.models import MLPModel
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# Create a master random key
key = jax.random.PRNGKey(0)

# Split it so one part initializes the network
key, init_key = jax.random.split(key)
    
config = load_config()
model = MLPModel(config,key)

def u_theta(params, x, t):
    """
    Define the PINN approximation u_theta(x,t).

    The network takes x and t as inputs and predicts a scalar u.
    """
    # Combine x and t into one input vector [x, t]
    inp = jnp.array([x, t])

    # Feed [x, t] through the MLP
    return model.mlp_forward(params, inp)


def u_t(params, x, t):
    """
    Compute partial derivative of u with respect to t.

    jax.grad differentiates a scalar-output function with respect
    to its argument. Here the variable is tt.
    """
    return jax.grad(lambda tt: u_theta(params, x, tt))(t)


def u_x(params, x, t):
    """
    Compute partial derivative of u with respect to x.
    """
    return jax.grad(lambda xx: u_theta(params, xx, t))(x)


def u_xx(params, x, t):
    """
    Compute second partial derivative with respect to x.

    We do this by differentiating u_x with respect to x again.
    """
    return jax.grad(lambda xx: u_x(params, xx, t))(x)


def pde_residual(params, x, t, alpha):
    """
    Heat equation residual:
        f(x,t) = u_t - alpha * u_xx

    If the PDE is satisfied exactly, this should be zero.
    """
    return u_t(params, x, t) - alpha * u_xx(params, x, t)

# Vectorize theta over batches of x and t values
# in_axes=(None, 0, 0) means:
# - params is shared (not batched)
# - x is batched along axis 0
# - t is batched along axis 0
u_model_vmap = jax.vmap(u_theta, in_axes=(None, 0, 0))

# Vectorize PDE residual the same way
pde_residual_vmap = jax.vmap(pde_residual, in_axes=(None, 0, 0))