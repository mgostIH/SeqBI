from typing import Callable, Tuple, Any, Optional
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Float, Int, UInt, Array, PRNGKeyArray

def differentiable_l2_norm(x : Float[Array, "dim"]) -> Float[Array, ""]:
    squared_norm = jnp.sum(x ** 2)
    return jnp.sqrt(squared_norm + 1e-6) # Lower doesn't work for fp16

class SmoothNorm(eqx.Module):
    scale : Float[Array, ""]
    
    def __init__(self, dim : Float[Array, ""], *, key : PRNGKeyArray = None):
        super().__init__()
        self.scale = jnp.sqrt(dim)

    def __call__(self, x : Float[Array, "dim"]) -> Float[Array, "dim"]:
        norm = differentiable_l2_norm(x)
        return self.scale * (x / (norm + 1.))
    
# Test smoothnorm is differentiable at 0
norm = SmoothNorm(3)
x = jnp.zeros(3)
print(eqx.filter_value_and_grad(lambda x: jnp.sum(norm(x)))(x))