from transformer import SmoothNorm, CausalTransformer
import jax
import jax.numpy as jnp
import equinox as eqx

# Test smooth norm has a gradient at the origin and not NaN
x = jnp.zeros(2)
smooth_norm = SmoothNorm(2)
y = eqx.filter_value_and_grad(lambda x: smooth_norm(x).sum())(x)
assert jnp.all(jnp.isfinite(y[1])) 

# Test the same in fp16
x = jnp.zeros(2, dtype=jnp.float16)
smooth_norm = SmoothNorm(2)
y = eqx.filter_value_and_grad(lambda x: smooth_norm(x).sum())(x)
assert jnp.all(jnp.isfinite(y[1]))

# Test whether the transformer builds and can call a forward pass
key = jax.random.PRNGKey(0)
model_dim = 8
seq_len = 10
transformer = CausalTransformer(
    model_dim=model_dim,
    num_heads=2,
    attn_dropout=0.1,
    num_layers=2,
    hidden_dim=4,
    key=key,
)
X = jnp.zeros((seq_len, model_dim))
Y = transformer(X, key=key)
assert Y.shape == (seq_len, model_dim)

# Test whether it's differentiable
y = eqx.filter_value_and_grad(lambda x: transformer(x, key=key).sum())(X)
assert jnp.all(jnp.isfinite(y[1]))

# Test whether differentiable wrt parameters
# The gradient is an entire pytree, so we can't check it directly
y = eqx.filter_value_and_grad(lambda t : t(X, key=key).sum())(transformer)



