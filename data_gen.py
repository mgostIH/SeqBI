import numpy as np

import numpy as np

def generate_markov_sequence(T, observation_flag=True):
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
    observations = []

    for _ in range(T):
        # Transition to next state
        current_state = np.random.choice(4, p=transition_matrix[current_state])

        # Decide whether to output the state or N based on the observation flag
        if observation_flag and np.random.rand() < 0.5:
            observations.append(state_labels[current_state])
        elif observation_flag:
            observations.append('N')
        else:
            # If observation flag is off, always output the state
            observations.append(state_labels[current_state])

    return observations

# Generate and print sequences with observation flag on and off for debugging
sequences_with_obs = [generate_markov_sequence(10) for _ in range(5)]
sequences_without_obs = [generate_markov_sequence(10, observation_flag=False) for _ in range(5)]

sequences_with_obs, sequences_without_obs


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
observations = [0, 1, 3, 4, 2]  # Example observation sequence (indices)

DP, best_path = viterbi_algorithm(state_transition_prob, observation_prob_matrix, observations, initial_distribution)
DP, best_path
