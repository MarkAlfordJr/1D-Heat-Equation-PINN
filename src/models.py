import jax
import jax.numpy as jnp

def init_mlp(layer_sizes, key):
    """
        Creates the Parameters needed for the NN forward pass

        Args: 
        - layers_size = layer dimension for the NN architecture
        - key = a random key per layer

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
        Forward passes the created Parameters in order to get a potential solution

        Args: 
        - params = array of weights and biases created from the NN architecture layers
        - x = input vector of size (2,) since the PDE is u(x,t)

        Returns:
        scalar output = u(x,t). the potential solution to PDE that goes into the Physics Residual
    """
      # 'a' stands for the current activation vector flowing through the network
    a = x

    # Iterate through each layer of the PARAMs array
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
