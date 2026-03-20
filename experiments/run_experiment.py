import yaml
from src.models import MLPModel
from src.trainer import Trainer
import jax
import jax.numpy as jnp
from src.utils import Utils
import optax

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

    
def run_experiment():
    #1. load up config data
    config = load_config()
    
    optimizer = optax.adam(config["optimizer_learning_rate"])
    
    # Create a master random key
    key = jax.random.PRNGKey(0)

    # Split it so one part initializes the network
    key, init_key = jax.random.split(key)

    # Initialize neural network parameters
    model = MLPModel(config, key)
    params = model.init_mlp()

    # Start the training loop
    utils = Utils(config, params)
    trainer = Trainer(model, params, optimizer)
    trainer.train_loop(key)
    print("running function")