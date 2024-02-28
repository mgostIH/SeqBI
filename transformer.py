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
    
class CausalTransformer(eqx.Module):
    model_dim : Int[Array, ""]
    num_heads : Int[UInt, ""]
    attn_dropout : Float[Array, ""]
    num_layers : Int[UInt, ""]
    hidden_dim : Int[Array, ""]

    def __init__(self, model_dim : Int[Array, ""], num_heads : Int[UInt, ""], attn_dropout : Float[Array, ""], num_layers : Int[UInt, ""], hidden_dim : Int[Array, ""], *, key : PRNGKeyArray = None):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.norm = SmoothNorm(model_dim)

        key_mlp, key_attn, key_linears = jax.random.split(key, 3)
        self.mlps = [eqx.nn.MLP(
            in_size=model_dim, 
            out_size=hidden_dim, 
            width_size=2*hidden_dim, 
            depth = 2,
            use_bias=False,
            key=key_mlp_i) for key_mlp_i in jax.random.split(key_mlp, num_layers)]
        self.attns = [eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=model_dim,
            dropout_p=attn_dropout,

            key=key_attn_i) for key_attn_i in jax.random.split(key_attn, num_layers)]
        
        # One linear for queries, one for keys, one for values
        self.linears = [[eqx.nn.Linear(in_features=model_dim, 
                                       out_features=model_dim, 
                                       use_bias=False, 
                                       key=key_linear_i) 
                                       for key_linear_i in jax.random.split(key_linear, 3)] 
                                       for key_linear in jax.random.split(key_linears, num_layers)]

        def __call__(x : Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
            for i in range(num_layers):
                x_res = x
                x = eqx.filter_vmap(self.norm)(x)
                q, k, v = [eqx.filter_vmap(linear)(x) for linear in self.linears[i]]
                mask = jnp.tril(jnp.ones((x.shape[0], x.shape[0])))
                attn = self.attns[i](q, k, v, mask=mask)
                mlp_out = eqx.filter_vmap(self.mlps[i])(attn)
                x = x_res + mlp_out + attn
            return x