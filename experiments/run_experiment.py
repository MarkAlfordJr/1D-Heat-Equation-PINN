import yaml
from src.model import MLPModel
from src.train import Trainer
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
    print("this is the learning rate: {}".format(config["optimizer_learning_rate"]))
    
    # Create a master random key
    master_key = jax.random.PRNGKey(0)

    # Split it so one part initializes the network
    key, init_key = jax.random.split(master_key)

    # Initialize neural network parameters
    model = MLPModel(config, init_key)
    params = model.init_mlp()

    # Start the training loop
    trainer = Trainer(model, params, optimizer, config)
    trainer.train_loop(key)
    print("running function")