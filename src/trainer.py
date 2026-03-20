import jax
import jax.numpy as jnp
import optax
from src.losses import loss_fn


# Create Adam optimizer
def make_train_step(model, optimizer, config):
    def train_step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss_fn(params, batch))(params, model, batch, config)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    return jax.jit(train_step, static_argnames=())


class Trainer:
    def __init__(self, model, params, optimizer):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.opt_state = optimizer.init(params)

        self.train_step = jax.jit(
            make_train_step,
            static_argnames=("model", "optimizer"),
        )

    def sample_training_points(self, key, N_f, N_bc, N_ic):
        """
        Sample training points for:
        1. PDE residual inside the domain
        2. Boundary conditions on x=0 and x=1
        3. Initial condition on t=0

        Args: 
            - N_f, N_bc, N_ic = collocation points described in problem 
            - key = random key to be split up
        """
        x_min, x_max = self.config["x_min"], self.config["x_max"]
        t_min, t_max = self.config["t_min"], self.config["t_max"]
        # Split one random key into four smaller keys
        # so each sampling step uses independent randomness
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # -----------------------------
        # Interior PDE collocation points
        # -----------------------------

        # Random x values inside [0,1]
        x_f = jax.random.uniform(k1, (N_f,), minval=x_min, maxval=x_max)

        # Random t values inside [0,1]
        t_f = jax.random.uniform(k2, (N_f,), minval=x_min, maxval=x_max)

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

    def train_loop(model, params, optimizer, dataset, config):
        opt_state = optimizer.init(params)
        train_step = make_train_step(model, optimizer, config)

        epochs = config["epochs"]
        loss_history = []

        for epoch in range(epochs):
            # batch must come from sample training points.
            # find where its 
            batch = dataset.sample_batch()

            params, opt_state, loss_value, aux = train_step(params, opt_state, batch)

            loss_history.append(loss_value)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"total={loss_value:.6f} | "
                    f"pde={aux['pde_loss']:.6f} | "
                    f"bc={aux['bc_loss']:.6f} | "
                    f"ic={aux['ic_loss']:.6f}"
                )

        return params, opt_state, loss_history




