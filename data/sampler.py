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