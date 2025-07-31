from jax.numpy import float32
from typing import Tuple
from typing import List
from models.dense_layer import DenseLayer
import jax
import jax.numpy as jnp
from jax import Array


class NeuralNetwork:
    def __init__(self, prng_key : int) -> None:
        self.layers : List[DenseLayer] = []
        self.prng_key = jax.random.PRNGKey(prng_key)
        self.trained = False
        
    def add(self, layer : DenseLayer)-> None:
        self.layers.append(layer)
    
    def __call__(self, x : Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x
        
    def get_params(self) -> List[Tuple[Array, Array]]:
        return [(layer.weight, layer.bias) for layer in self.layers]

    def predict(self, x : Array) -> Array | float32:
        if self.trained: 
            return self(x)
        else: 
            raise RuntimeError("model is not trained yet")

    
