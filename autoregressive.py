# The following code will provide a supporting architecture and loss for any autoregressive model
# We consider models of the form:
# Inputs : Tensor of shape (n, d)
# Outputs: Tensor of shape (n, d)
# where n is the sequence length and d is the dimension of the input/output
# In general we consider only causal models, i.e. the output at time t can only depend on the input at time t or before
# The surrounding architecture will embed the input tokens, and then apply the autoregressive model to the sequence of embeddings
# The loss will be the crossentropy loss between the predicted token and the actual token
# Batches are handled by vmap so we don't need to worry about them

from typing import Callable, Tuple, Any, Optional
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Float, Int, UInt, Array, PRNGKeyArray

# The autoregressive model is a function that takes in a sequence of tokens and outputs a sequence of tokens
AutoregressiveModel = Callable[[Float[Array, "seq_len model_dim"]], Float[Array, "seq_len model_dim"]]

# The loss is a cross entropy we need to implement ourselves, the target is always a one-hot vector, we work with logits
# The loss defined here only considers one token at a time, so we need to sum it over the sequence length for the tokens that matter
# The target is a number representing at what position of the one-hot vector the 1 is, so it's a number between 0 and logits_dim - 1
def stable_single_cross_entropy(y_pred: Float[Array, "logits_dim"], target: UInt[Array, ""]) -> Float[Array, ""]:
    # 'a' is the maximum logit to stabilize the computation
    a = jnp.max(y_pred)
    # Applying the log-sum-exp trick for numerical stability
    log_sum_exp = a + jnp.log(jnp.sum(jnp.exp(y_pred - a)))
    # Computing the cross-entropy
    return -y_pred[target] + log_sum_exp

# Keep this in vector form since a caller will again use vmap to apply it to a batch and then sum to get the total loss
sequence_cross_entropy = jax.vmap(stable_single_cross_entropy, in_axes=(0, 0))

# A model that embeds the input tokens, applies the autoregressive model and projects each vector to a logits vector
class CompleteAutoregressiveModel(eqx.Module):
    embedding : eqx.nn.Embedding
    autoregressive_model : AutoregressiveModel
    projection : eqx.nn.Linear
    model_dim : int
    logits_dim : int

    def __init__(self, autoregressive_model : AutoregressiveModel, logits_dim : int, model_dim : int, *, key : PRNGKeyArray):
        super().__init__()
        embedding_key, linear_key = jax.random.split(key, 2)
        self.embedding = eqx.nn.Embedding(num_embeddings=logits_dim, embedding_size=model_dim, key=embedding_key)
        self.autoregressive_model = autoregressive_model
        self.projection = eqx.nn.Linear(in_features=model_dim, out_features=logits_dim, use_bias=False, key=linear_key)
        self.model_dim = model_dim
        self.logits_dim = logits_dim

    def __call__(self, x : Int[Array, "seq_len"]) -> Float[Array, "seq_len logits_dim"]:
        # Embedding the input tokens
        x = jax.vmap(self.embedding)(x)
        # Applying the autoregressive model
        x = self.autoregressive_model(x)
        # Projecting each vector to a logits vector
        x = jax.vmap(self.projection)(x)
        return x
    
    def embed_no_logits(self, x : Int[Array, "seq_len"]) -> Float[Array, "seq_len model_dim"]:
        # Embedding the input tokens
        x = jax.vmap(self.embedding)(x)
        # Applying the autoregressive model
        x = self.autoregressive_model(x)
        return x

    def simple_cross_entropy_loss_on_tokens(self, x : Int[Array, "seq_len"]) -> Float[Array, ""]:
        input = prepare_for_autoregressive_model(x)
        # Computing the logits
        logits = self(input)
        # Computing the loss
        loss = sequence_cross_entropy(logits, x)
        # Summing the loss over the sequence length
        return jnp.sum(loss)
    
# A helper function that takes a sequence of tokens to predict, prepends a start token and removes the last token
# This is useful for training since we want to predict the next token given the previous tokens
# The start token is by convention -1
def prepare_for_autoregressive_model(x : Int[Array, "seq_len"]) -> Int[Array, "seq_len"]:
    return jnp.concatenate([jnp.array([-1]), x[:-1]])
