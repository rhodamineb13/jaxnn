from jax.nn import tanh
from jax.nn import sigmoid
from jax.numpy import float32
from typing import Callable
from models.dense_layer import DenseLayer
from models.neural import NeuralNetwork
from jax import Array
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import math
from utils.spring_solution import init_spring
from utils.loss_functions import MSELoss


SEED_NUMBER : int = 42
k : float = 3.0
b : float = 2.5
m : float = 10
amp : float = 1.0
phase : float = 0.0

def main() -> None:
    SEED : Array  = jax.random.PRNGKey(SEED_NUMBER)
    x : Array = jnp.linspace(1, 5, 1000)
    y : Array = init_spring(x, k, b, m, amp, 0.0)

    out : Array = MSELoss(x, y)


    nn : NeuralNetwork = NeuralNetwork(SEED_NUMBER)
    
    nn.add(DenseLayer(x.shape[-1], 64, "normal", tanh))
    nn.add(DenseLayer(64, 64, "normal", tanh))
    nn.add(DenseLayer(64, 64, "normal", tanh))
    nn.add(DenseLayer(64, 1, "normal", tanh))

    

    return

if __name__ == "__main__":
    main()