import jax
import jax.numpy as jnp

class MLPModel():
    def __init__(self, config, key):
        self.config = config
        self.key = key
    
    def init_mlp(self):
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
        keys = jax.random.split(self.key, len(self.config["model_layer_size"]) - 1)

        # Loop through each pair of consecutive layer sizes
        # Example: (2 -> 64), (64 -> 64), (64 -> 64), (64 -> 1)
        for k, (in_dim, out_dim) in zip(keys, zip(self.config["model_layer_size"][:-1], self.config["model_layer_size"][1:])):
            limit = jnp.sqrt(6.0 / (in_dim + out_dim))
            Weights = jax.random.uniform(k, (in_dim, out_dim), minval=-limit, maxval=limit)
            biases = jnp.zeros((out_dim,))

            # Store this layer's parameters
            params.append((Weights, biases))

        return params
    
    def mlp_forward(self, params, x):
        """
            Forward passes the created Parameters in order to get a potential solution

            Args: 
            - params = array of weights and biases created from the NN architecture layers
            - x = input vector of size (2,) since the PDE is u(x,t)

            Returns:
            scalar output = u(x,t). the potential solution to PDE that goes into the Physics Residual
        """
        # 'a' stands for the current activation vector flowing through the network
        activation_vector = x

        # Iterate through each layer of the PARAMs array
        for item, (Weight, biases) in enumerate(params):
            # Compute affine transformation z = aW + b
            affine_transformation = jnp.dot(activation_vector, Weight) + biases

            # For all hidden layers, apply tanh activation
            if item < len(params) - 1:
                activation_vector = jnp.tanh(affine_transformation)
            else:
                # For the final layer, use no activation
                # because we want a raw scalar output
                activation_vector = affine_transformation

        # Final output has shape (1,), so take the first element
        return activation_vector[0]