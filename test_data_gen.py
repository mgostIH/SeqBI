from data_gen import generate_markov_sequence, viterbi_algorithm, w
import numpy as np

# Generate and print sequences with observation flag on and off for debugging
sequences_with_obs = [generate_markov_sequence(10) for _ in range(5)]
sequences_without_obs = [generate_markov_sequence(10, observation_flag=False) for _ in range(5)]

sequences_with_obs, sequences_without_obs

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
observations = [4, 4, 2]  # Example observation sequence (indices)

DP, best_path = viterbi_algorithm(state_transition_prob, observation_prob_matrix, observations, initial_distribution)
DP, best_path

# The above has been tested manually and is correct