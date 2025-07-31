from jax import Array
import jax.numpy as jnp

def init_spring(x : Array, k : float, b : float, m : float, amp : float, phase : float) -> Array:
    if b**2 - 4*k*m >= 0:
        raise ValueError("b**2 - 4*k*m must be less than zero")
    omega : Array = jnp.sqrt(4*k*m - b**2)/(2*m)
    y : Array = amp * jnp.exp(-x * b/2*m) * jnp.sin(omega * x)
    return y