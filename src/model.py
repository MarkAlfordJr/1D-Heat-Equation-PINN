import jax
import jax.numpy as jnp

def init_mlp(layer_sizes, key):
    """
    Produces the weight and biases parameters the MLP pass forward.

    Args:
        layer_sizes (list): 
        key (int): random key needed for MLP model

    Returns:
        params: [[W1,b1], [W2,b2], [Wn,bn]]
    """
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)

    for k, (in_dim, out_dim) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        W = jax.random.uniform(k, (in_dim, out_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim,))
        params.append((W, b))

    return params

def mlp_forward(params, x):
    a = x
    for i, (W, b) in enumerate(params):
        z = jnp.dot(a, W) + b
        if i < len(params) - 1:
            a = jnp.tanh(z)
        else:
            a = z
    return a[0]

def u_model(params, x, t):
    inp = jnp.array([x,t])
    return mlp_forward(params, inp)