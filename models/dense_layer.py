from typing import Callable
from jax import Array
import jax
import jax.numpy as jnp

class DenseLayer:
    def __init__(self, in_dim : int, out_dim : int, initialization : str, activation : Callable | None = None) -> None:
        self.in_dim : int = in_dim
        self.out_dim : int = out_dim
        self.prng_key : Array = jax.random.PRNGKey(42)
        match initialization:
            case "normal":
                self.weight = jax.random.normal(self.prng_key, shape=[in_dim, out_dim])
            case "uniform":
                self.weight = jax.random.uniform(self.prng_key, shape=[in_dim, out_dim])
            case _:
                raise ValueError(f"unknown initialization: {initialization}")
        self.bias = jnp.zeros(shape=out_dim)
        self.activation = activation


    def __call__(self, x : Array) -> Array:
        if x.dtype != jnp.float32:
            x = x.astype(jnp.float32)
        out : Array = x @ self.weight + self.bias
        if self.activation:
            out = self.activation(out)
        return out
    

