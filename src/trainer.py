import jax
import optax
from losses import loss_fn
from config import LAYER_SIZES, learning_rate, epochs, N_f, N_bc, N_ic, print_every
from utils import sample_training_points
from models import init_mlp
from trainer import optimizer, train_step

# Create Adam optimizer
optimizer = optax.adam(learning_rate)


@jax.jit
def train_step(params, opt_state, batch):
    """
    One training step:
    1. Compute loss
    2. Compute gradients of loss wrt params
    3. Update parameters with Adam
    """

    # value_and_grad computes both:
    # - the loss value
    # - the gradient of that loss wrt params
    #
    # has_aux=True means loss_fn returns:
    #   (main_loss, extra_info)
    # where extra_info here is (loss_pde, loss_bc, loss_ic)
    (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)

    # Use optimizer to turn gradients into parameter updates
    updates, opt_state = optimizer.update(grads, opt_state, params)

    # Apply updates to parameters
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss_value, aux

def train_loop():
    # Create a master random key
    key = jax.random.PRNGKey(0)

    # Split it so one part initializes the network
    key, init_key = jax.random.split(key)

    # Initialize neural network parameters
    params = init_mlp(LAYER_SIZES, init_key)

    # Initialize optimizer state for Adam
    opt_state = optimizer.init(params)

    # Store loss values for plotting later
    loss_history = []

    # Main training loop
    for epoch in range(1, epochs + 1):
        # Split random key each epoch so we can resample fresh points
        key, batch_key = jax.random.split(key)

        # Sample a new batch of collocation, boundary, and initial points
        batch = sample_training_points(batch_key, N_f, N_bc, N_ic)

        # Perform one optimization step
        params, opt_state, loss_value, aux = train_step(params, opt_state, batch)

        # Unpack the separate loss terms
        loss_pde, loss_bc, loss_ic = aux

        # Save total loss for plotting
        loss_history.append(float(loss_value))

        # Print progress every print_every epochs
        if epoch % print_every == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Total: {loss_value:.6e} | "
                f"PDE: {loss_pde:.6e} | "
                f"BC: {loss_bc:.6e} | "
                f"IC: {loss_ic:.6e}"
            )
        
    return loss_history

