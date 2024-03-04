import numpy as np

import jax.numpy as jnp
import jax

def generate_markov_sequence(T, observation_flag=True, seed=None):
    # Define the transition matrix for the Markov chain
    # Rows are current states, columns are next states: [A, B, C, F]
    transition_matrix = np.array([
        [0.6, 0.4, 0.0, 0.0],  # A -> A, B
        [0.6, 0.0, 0.4, 0.0],  # B -> A, C
        [0.0, 0.6, 0.0, 0.4],  # C -> B, F
        [0.0, 0.0, 0.0, 1.0]   # F -> F
    ])

    # Initialize the sequence
    state_labels = ['A', 'B', 'C', 'F']
    current_state = 0  # Start at state A
    states = []

    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed)

    for _ in range(T):
        states.append(state_labels[current_state])
        # Transition to next state
        current_state = rng.choice(4, p=transition_matrix[current_state])
        

    # Determine whether to swap the state with 'N' based on 50% probability
    if observation_flag:
        for i in range(len(states)):
            if rng.random() < 0.5:
                states[i] = 'N'

    return states

def jax_generate_markov_sequence(T, key):
    gen_key, hide_key = jax.random.split(key)
    states = jax_generate_markov_states(T, gen_key)
    return jax_hide_states(states, hide_key)

# Assume filter is a function that takes a single row, a key and returns true or false
def jax_generate_filtered_markov_matrix(T, N, filter, key):
    gen_key, hide_key = jax.random.split(key)
    hide_keys = jax.random.split(hide_key, N)
    generated_states = jnp.zeros((0, T), dtype=jnp.int8)
    while generated_states.shape[0] < N:
        gen_key, filter_key = jax.random.split(gen_key)
        gen_keys = jax.random.split(gen_key, N)
        filter_keys = jax.random.split(filter_key, N)
        states = jax.vmap(jax_generate_markov_states, in_axes=(None, 0))(T, gen_keys)
        row_mask = jax.vmap(filter, in_axes=(0, 0))(states, filter_keys) # (N,)
        generated_states = jnp.vstack([generated_states, states[row_mask]])

    return jax.vmap(jax_hide_states, in_axes=(0, 0))(generated_states[:N], hide_keys)

def jax_generate_markov_states(T, key):
    # Define the transition matrix for the Markov chain
    # Rows are current states, columns are next states: [A, B, C, F]
    transition_matrix = jnp.array([
        [0.6, 0.4, 0.0, 0.0],  # A -> A, B
        [0.6, 0.0, 0.4, 0.0],  # B -> A, C
        [0.0, 0.6, 0.0, 0.4],  # C -> B, F
        [0.0, 0.0, 0.0, 1.0]   # F -> F
    ])

    # Initialize the sequence
    state_labels = jnp.array([0, 1, 2, 3])
    current_state = 0  # Start at state A
    states = []

    for _ in range(T):
        choice_key, key = jax.random.split(key)
        states.append(state_labels[current_state])
        # Transition to next state
        current_state = jax.random.choice(choice_key, 4, p=transition_matrix[current_state])
    return jnp.array(states)

def jax_hide_states(states, key):
    mask = jax.random.bernoulli(key, 0.5, shape=states.shape)
    return jnp.where(mask, 4, states)


def w(p):
    """Convert probability to weight."""
    if p == 0:
        return np.inf
    else:
        return -np.log(p)

def viterbi_algorithm(state_transition_prob, observation_prob_matrix, observations, initial_distribution):
    S = state_transition_prob.shape[0]  # Number of states
    T = len(observations)  # Length of observation sequence
    O = observation_prob_matrix.shape[1]  # Number of observations

    # Convert probabilities to weights
    state_transition_weight = np.vectorize(w)(state_transition_prob)
    observation_weight_matrix = np.vectorize(w)(observation_prob_matrix)

    # Initialize matrices
    DP = np.full((S, T+1), np.inf)  # Cumulative weights matrix, initialized to inf
    path = np.zeros((S, T+1), dtype=int)  # To store the path of most likely states
    
    # Initialize the first column of DP with the initial distribution converted to weights
    DP[:, 0] = np.vectorize(w)(initial_distribution)

    # Viterbi algorithm
    for t in range(1, T+1):
        observation_index = observations[t-1]  # Get the current observation index
        for j in range(S):
            # Compute the cumulative weight for each state at time t
            for i in range(S):
                # Transition weight includes the weight of observing O given current state i and transition to j
                transition_weight = observation_weight_matrix[i, observation_index] + state_transition_weight[i, j]
                cumulative_weight = DP[i, t-1] + transition_weight
                
                if cumulative_weight < DP[j, t]:
                    DP[j, t] = cumulative_weight
                    path[j, t] = i

    # Find the ending state with the smallest cumulative weight
    end_state = np.argmin(DP[:, T])
    best_path = [end_state]

    # Trace back the best path
    for t in range(T, 0, -1):
        end_state = path[end_state, t]
        best_path.insert(0, end_state)

    return DP, best_path

