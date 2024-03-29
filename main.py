from transformer import CausalTransformer
import autoregressive as ar
import optax
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import data_gen
import json

# Parameters of the environment
setup_parameters = {
    "sequences" : 100_000, # Number of sequences to generate
    "seq_len" : 20, # Length of each sequence 
    "vocab_size" : 5+1, # Size of the vocabulary (5 observations + start token)
    "test_size" : 0.2, # Size of the test set
    "rng_seed" : 1337, # Random number generator seed
}

training_parameters = {
    "learning_rate" : 3e-4, # Learning rate
    "epochs" : 100, # Number of epochs
    "batch_size" : 1000, # Batch size
    "optimizer" : optax.adam, # Optimizer
}

model_parameters = {
    "model_dim" : 128, # Dimension of the model
    "num_heads" : 4, # Number of heads in the transformer
    "attn_dropout" : 0.0, # Dropout in the transformer
    "num_layers" : 3, # Number of layers in the transformer
    "hidden_dim" : 128, # Hidden dimension in the transformer
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


def generate_sequences(seq_amount, seq_len, markov_key):
    seq_keys = jax.random.split(markov_key, seq_amount)
    sequences = eqx.filter_vmap(data_gen.jax_generate_markov_sequence, in_axes=(None, 0))(seq_len, seq_keys)
    return sequences

def loss_fn(model, batch, dropout_key):
    f = lambda x, key : model.simple_cross_entropy_loss_on_tokens(x, key=key)
    losses = eqx.filter_vmap(f, in_axes=(0, None))(batch, dropout_key)
    return jnp.mean(losses)




def training_loop(model, opt_state, optimizer, train_sequences, training_parameters, key):
    @eqx.filter_jit
    def train_step(model, opt_state, batch, dropout_key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, dropout_key)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss
    try:
        for epoch in range(training_parameters["epochs"]):
            for i in range(0, len(train_sequences), training_parameters["batch_size"]):
                dropout_key, key = jax.random.split(key)
                batch = train_sequences[i:i+training_parameters["batch_size"]]
                model, opt_state, loss = train_step(model, opt_state, batch, dropout_key)
            print(f"Epoch {epoch} - Loss: {loss}")
    except KeyboardInterrupt:
        pass
    return model, opt_state, loss


key = jax.random.PRNGKey(setup_parameters["rng_seed"])
later_key, key = jax.random.split(key) # Used to make saved and trained model rng the same after try
# Check if the model is saved and load it if it is
try:
    model, model_parameters = load_model("model.eqx")
    model_filtered, _ = load_model("model_filtered.eqx")
    print("Model loaded")
except FileNotFoundError:
    init_key, init_key_2, key = jax.random.split(key, 3)
    model, opt_state, optimizer = initialize_model(setup_parameters, model_parameters, training_parameters, init_key)
    model_filtered, opt_state_filtered, optimizer_filtered = initialize_model(setup_parameters, model_parameters, training_parameters, init_key_2)
    print("Model initialized")
    markov_key, key = jax.random.split(key)
    train_sequences = generate_sequences(setup_parameters["sequences"], setup_parameters["seq_len"], markov_key)
    
    filter = lambda x, key : ~jnp.any(x[:10] == 3)
    key, filter_key = jax.random.split(key)
    filtered_train_sequences = data_gen.jax_generate_filtered_markov_matrix(
        setup_parameters["seq_len"], 
        setup_parameters["sequences"], 
        filter, 
        filter_key)
    
    print("Data generated")
    train_key, train_key_filtered, key = jax.random.split(key, 3)
    model, _, _ = training_loop(model, opt_state, optimizer, train_sequences, training_parameters, train_key)
    print("Original model trained")
    model_filtered, _, _ = training_loop(model_filtered, opt_state_filtered, optimizer_filtered, filtered_train_sequences, training_parameters, train_key_filtered)
    print("Filtered model trained")
    save_model(model, "model.eqx")
    save_model(model_filtered, "model_filtered.eqx")
    print("Model saved")

key = later_key
#model = eqx.nn.inference_mode(model)

# Generate a test sequence
test_key, key = jax.random.split(key)
test_sequences = generate_sequences(10, setup_parameters["seq_len"], test_key)
test_sequences = eqx.filter_vmap(ar.prepare_for_autoregressive_model)(test_sequences)

print("Test sequences generated")
print("Test sequences:", test_sequences)

test_sequence = test_sequences[0]

test_sequence = jnp.array([-1, 0, 1, 0, 1, 4, 4], dtype=jnp.int8)
# Calculate logits for the test sequence
logits = model(test_sequence, key=test_key)
jnp.set_printoptions(suppress=True, precision=3)
print(f"Logits for the test sequence: \n{logits}")

probabilities = eqx.filter_vmap(jax.nn.softmax)(logits)
print(f"Probabilities for the test sequence: \n{probabilities}")

import matplotlib.pyplot as plt
import seaborn as sns

# Generating a (20, 5) tensor with random probabilities
# Each row will sum to 1
probabilities = probabilities[:, :-1]

# Column labels
columns = ['A', 'B', 'C', 'F', 'N']

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(probabilities, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=columns)

plt.title('Probability Distribution Heatmap')
plt.ylabel('Sample Index')
plt.xlabel('Object Label')

# Show the plot
plt.show()


