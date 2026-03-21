import jax
import jax.numpy as jnp
import optax
from functools import partial
from src.loss import loss_fn


"""
    Process for training PINNs
    1. sample points from the domain
    2. compute the prediction (u_theta)
    3. compute the PDE residual
    4. compute the loss fucntion (residual loss, bc loss, ic loss)
    5. differentiate total loss
    6. update parameters with an optimizer
    7. repeat this over epochs

    key methods
    - sample collocation point()
    - train step
        - evaluate the loss and compute gradient(value and grad)
        - apply optimizier update (optimizier.update)
        - return new params and optimizer state
    - training loop

    training flow
    a. sample collocaation points
    b. compute the network outputs (u_theta)
    c. compute residual
    d. compute loss function
    e. compute gradient
    f. optimize the parameters taken from the NN
    g. repeat over epochs
"""

# Create Adam optimizer

class Trainer:
    def __init__(self, model, params, optimizer, config):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.opt_state = optimizer.init(params)
        self.config = config

    def sample_training_points(self, key, N_f, N_bc, N_ic):
        x_min, x_max = self.config["x_min"], self.config["x_max"]
        t_min, t_max = self.config["t_min"], self.config["t_max"]
    
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Interior Points: random values for x and t inside sptial domain of [0,1]
        x_f = jax.random.uniform(k1, (N_f,), minval=x_min, maxval=x_max)
        t_f = jax.random.uniform(k2, (N_f,), minval=x_min, maxval=x_max)

        # Boundary Condition Points: Random times for boundary enforcement
        t_bc = jax.random.uniform(k3, (N_bc,), minval=t_min, maxval=t_max)
        # Left boundary x=0
        x_bc_left = jnp.zeros((N_bc,))
        # Right boundary x=1
        x_bc_right = jnp.ones((N_bc,))

        # Initial Condition: Random x positions along the initial line t=0
        x_ic = jax.random.uniform(k4, (N_ic,), minval=x_min, maxval=x_max)
        # Initial condition occurs at t=0
        t_ic = jnp.zeros((N_ic,))

        return x_f, t_f, x_bc_left, x_bc_right, t_bc, x_ic, t_ic
    
    # Create Adam optimizer
    # optimizer = optax.adam(learning_rate)

    @partial(jax.jit, static_argnums=0)
    def train_step(self, params, opt_state, batch):
        (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        print("PARAM TREE:")
        print(jax.tree.map(lambda x: (type(x), getattr(x, "shape", None), getattr(x, "dtype", None)), params))

        print("GRAD TREE:")
        print(jax.tree.map(lambda x: (type(x), getattr(x, "shape", None), getattr(x, "dtype", None)), grads))
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, aux
    

    def train_loop(self, key):
        loss_history = []
        key, batch_key = jax.random.split(key)
        for epoch in range(1, self.config["epochs"]+1):
            batch = self.sample_training_points(batch_key, self.config["pde_res_colloc_points"], self.config["bc_colloc_points"], self.config["ic_colloc_points"])
            self.params, self.opt_state, loss_value, aux = self.train_step(self.params, self.opt_state, batch)
            loss_pde, loss_bc, loss_ic = aux
            loss_history.append(float(loss_value))
            if epoch % self.config["print_every_epoch"] == 0:
                 print(
            f"Epoch {epoch:5d} | "
            f"Total: {loss_value:.6e} | "
            f"PDE: {loss_pde:.6e} | "
            f"BC: {loss_bc:.6e} | "
            f"IC: {loss_ic:.6e}"
        )



