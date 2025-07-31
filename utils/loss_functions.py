from jax.numpy import float32
import jax
import jax.numpy as jnp
from jax import Array

def MSELoss(y_true : Array, y_hat : Array) -> Array:
    return jnp.mean(jnp.sum((y_true - y_hat)**2))

def logcoshloss(y_true : Array, y_hat: Array) -> float32:
    return jnp.mean(jnp.sum(jnp.log(jnp.cosh((y_true - y_hat)))))