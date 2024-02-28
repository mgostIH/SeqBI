from transformer import CausalTransformer
import autoregressive as ar
import optax
import jax
import jax.numpy as jnp
import equinox as eqx
import data_gen
import json

# Parameters of the environment
setup_parameters = {
    "sequences" : 1_000, # Number of sequences to generate
    "seq_len" : 20, # Length of each sequence 
    "vocab_size" : 5, # Size of the vocabulary
    "test_size" : 0.2, # Size of the test set
    "rng_seed" : 1337, # Random number generator seed
}

training_parameters = {
    "learning_rate" : 3e-4, # Learning rate
    "epochs" : 100, # Number of epochs
    "batch_size" : 128, # Batch size
    "optimizer" : optax.adam, # Optimizer
}

model_parameters = {
    "model_dim" : 32, # Dimension of the model
    "num_heads" : 4, # Number of heads in the transformer
    "attn_dropout" : 0.0, # Dropout in the transformer
    "num_layers" : 4, # Number of layers in the transformer
    "hidden_dim" : 32, # Hidden dimension in the transformer
}


def save_model(model, path):
    with open(path, "wb") as f:
        model_parameters_str = json.dumps(model_parameters)
        f.write((model_parameters_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load_model(path):
    with open(path, "rb") as f:
        model_parameters_str = f.readline().decode()
        saved_model_parameters = json.loads(model_parameters_str)
        transformer = CausalTransformer(**saved_model_parameters, key=jax.random.PRNGKey(0))
        model = ar.CompleteAutoregressiveModel(transformer, 
                                               setup_parameters["vocab_size"], 
                                               saved_model_parameters["model_dim"], 
                                               key=jax.random.PRNGKey(0))
        model = eqx.tree_deserialise_leaves(f, model)
    return model, model_parameters


def initialize_model(setup_parameters, model_parameters, training_parameters, key):
        transformer_key, key = jax.random.split(key)
        transformer = CausalTransformer(
            **model_parameters,
            key=transformer_key,
        )
        model_key, key = jax.random.split(key)
        model = ar.CompleteAutoregressiveModel(transformer, 
                                            setup_parameters["vocab_size"], 
                                            model_parameters["model_dim"], 
                                            key=model_key)

        optimizer = training_parameters["optimizer"](training_parameters["learning_rate"]) # Adam
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        
        return model, opt_state, optimizer


def generate_sequences(setup_parameters, markov_key):
    # Generate data
    # Setup an empty matrix of size (sequences, seq_len) to store the sequences
    # Use int8 to save memory
    sequences = jnp.zeros((setup_parameters["sequences"], setup_parameters["seq_len"]), dtype=jnp.int8)
    # Generate the sequences
    for i in range(setup_parameters["sequences"]):
        sequence_seeds = jax.random.split(markov_key, setup_parameters["sequences"])
        sequence = data_gen.generate_markov_sequence(setup_parameters["seq_len"], seed=int(sequence_seeds[i][0]))
        # A sequence here is made of letters A, B, C, F, N, convert them to 0 1 2 3 4
        letter_map = {"A" : 0, "B" : 1, "C" : 2, "F" : 3, "N" : 4}
        sequence = [letter_map[letter] for letter in sequence]
        sequence = jnp.array(sequence, dtype=jnp.int8)
        sequences = sequences.at[i].set(sequence)
    # Split the data into training and test
    test_size = int(setup_parameters["sequences"] * setup_parameters["test_size"])
    train_sequences = sequences[:-test_size]
    test_sequences = sequences[-test_size:]
    return train_sequences, test_sequences

def loss_fn(model, batch, dropout_key):
    f = lambda x, key : model.simple_cross_entropy_loss_on_tokens(x, key=key)
    losses = eqx.filter_vmap(f, in_axes=(0, None))(batch, dropout_key)
    return jnp.mean(losses)

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, dropout_key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, dropout_key)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


def training_loop(model, opt_state, optimizer, train_sequences, training_parameters, key):
    try:
        for epoch in range(training_parameters["epochs"]):
            for i in range(0, len(train_sequences), training_parameters["batch_size"]):
                dropout_key, key = jax.random.split(key)
                batch = train_sequences[i:i+training_parameters["batch_size"]]
                model, opt_state, loss = train_step(model, opt_state, optimizer, batch, dropout_key)
            print(f"Epoch {epoch} - Loss: {loss}")
    except KeyboardInterrupt:
        return


key = jax.random.PRNGKey(setup_parameters["rng_seed"])

# Check if the model is saved and load it if it is
try:
    model, model_parameters = load_model("model.eqx")
    print("Model loaded")
except FileNotFoundError:
    init_key, key = jax.random.split(key)
    model, opt_state, optimizer = initialize_model(setup_parameters, model_parameters, training_parameters, init_key)
    print("Model initialized")
    markov_key, key = jax.random.split(key)
    train_sequences, test_sequences = generate_sequences(setup_parameters, markov_key)
    train_key, key = jax.random.split(key)
    training_loop(model, opt_state, optimizer, train_sequences, training_parameters, train_key)
    save_model(model, "model.eqx")
    print("Model saved")

# Usage:

