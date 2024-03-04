from data_gen import *
import numpy as np
import jax
import jax.numpy as jnp


seq_amount = 30
seq_len = 20

# Generate and print sequences with observation flag on and off for debugging
sequences_with_obs = [generate_markov_sequence(seq_len, seed=i) for i in range(seq_amount)]
sequences_without_obs = [generate_markov_sequence(seq_len, seed=i, observation_flag=False) for i in range(seq_amount)]

sequences_with_obs, sequences_without_obs

# Test: Check whether sequences_with_obs is the same as sequences_without_obs except for the 'N' tokens
for i in range(seq_amount):
    for j in range(seq_len):
        if sequences_with_obs[i][j] != 'N':
            assert sequences_with_obs[i][j] == sequences_without_obs[i][j]

# Example usage
S = 4  # Number of states
O = 5  # Number of observations including 'N'

# Example state transition probabilities and observation probabilities
state_transition_prob = np.array([
    [0.6, 0.4, 0.0, 0.0],
    [0.6, 0.0, 0.4, 0.0],
    [0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 1.0]
])

observation_prob_matrix = np.array([
    [0.5, 0, 0, 0, 0.5],  # State A observation probabilities
    [0, 0.5, 0, 0, 0.5],  # State B
    [0, 0, 0.5, 0, 0.5],  # State C
    [0, 0, 0, 0.5, 0.5]     # State F
])

initial_distribution = np.array([1, 0, 0, 0])
observations = [0, 1, 0, 1, 4, 4, 4]  # Example observation sequence (indices)

DP, best_path = viterbi_algorithm(state_transition_prob, observation_prob_matrix, observations, initial_distribution)
DP, best_path

# The above has been tested manually and is correct

# Sample sequences using jax
key = jax.random.PRNGKey(0)
seq_keys = jax.random.split(key, seq_amount)
jax_sequences_with_obs = jax.vmap(jax_generate_markov_sequence, in_axes=(None, 0))(seq_len, seq_keys)

key, _ = jax.random.split(key)
filter = lambda x, key : ~jnp.any(x[:10] == 3)
jax_sequences_with_obs_filtered = jax_generate_filtered_markov_matrix(seq_len, seq_amount, filter, key)