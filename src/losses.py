import jax
import jax.numpy as jnp
from src.physics import pde_residual_vmap, u_model_vmap

def loss_fn(params, batch):
    """
    Compute total PINN loss.

    batch contains:
      x_f, t_f         -> interior PDE points
      x_bc_left/right  -> boundary x values
      t_bc             -> boundary times
      x_ic, t_ic       -> initial-condition points
    """
    x_f, t_f, x_bc_left, x_bc_right, t_bc, x_ic, t_ic = batch

    # -----------------------------
    # PDE residual loss
    # -----------------------------

    # Evaluate PDE residual at all interior collocation points
    f_vals = pde_residual_vmap(params, x_f, t_f)

    # Mean squared PDE residual
    # If residual is small, network approximately satisfies PDE
    loss_pde = jnp.mean(f_vals ** 2)

    # -----------------------------
    # Boundary condition loss
    # -----------------------------

    # Predict u at left boundary x=0
    u_left = u_model_vmap(params, x_bc_left, t_bc)

    # Predict u at right boundary x=1
    u_right = u_model_vmap(params, x_bc_right, t_bc)

    # Since boundary condition is u=0 at both ends,
    # we penalize squared deviation from zero
    loss_bc = jnp.mean(u_left ** 2) + jnp.mean(u_right ** 2)

    # -----------------------------
    # Initial condition loss
    # -----------------------------

    # Network prediction at t=0
    u_ic_pred = u_model_vmap(params, x_ic, t_ic)

    # True initial condition is sin(pi x)
    u_ic_true = jnp.sin(jnp.pi * x_ic)

    # Penalize mismatch with initial condition
    loss_ic = jnp.mean((u_ic_pred - u_ic_true) ** 2)

    # -----------------------------
    # Total loss
    # -----------------------------

    # Standard PINN total loss = PDE + boundary + initial
    total = loss_pde + loss_bc + loss_ic

    # Return total plus the individual parts for monitoring
    return total, (loss_pde, loss_bc, loss_ic)
