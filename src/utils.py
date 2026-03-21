import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.physics import u_model_vmap

class Utils():
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def exact_solution(self, x, t):
        """
        Known analytical solution for:
            u_t = alpha u_xx
            u(0,t)=0
            u(1,t)=0
            u(x,0)=sin(pi x)

        This is NOT used in training.
        It is only used after training to check how close the PINN is.
        """
        return jnp.exp(-(jnp.pi ** 2) * self.config["alpha"] * t) * jnp.sin(jnp.pi * x)

    def plot_results(self):
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
            u_pred = u_model_vmap(self.params, x_plot, t_array)

            # Exact solution for comparison
            u_true = self.exact_solution(x_plot, t_array)

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



