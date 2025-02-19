import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabilities of hidden states
            transition_p (np.ndarray): transition probabilities between hidden states
            emission_p (np.ndarray): emission probabilities from transition to hidden states
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        self.n_states = len(hidden_states)

        # Assertions to ensure correct probabilities
        assert np.allclose(np.sum(self.prior_p), 1), "Prior probabilities must sum to 1."
        assert np.allclose(np.sum(self.transition_p, axis=1), 1), "Each row in transition_p must sum to 1."
        assert np.allclose(np.sum(self.emission_p, axis=1), 1), "Each row in emission_p must sum to 1."

        # Ensure at least one hidden state has a nonzero prior probability
        assert np.any(self.prior_p > 0), "At least one hidden state must have a nonzero prior probability."

    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """
        input_observation_states = np.atleast_1d(input_observation_states)

        # edge cases
        if input_observation_states.size == 0:
            raise ValueError("Input sequence is empty.")

        for state in input_observation_states:
            if state not in self.observation_states_dict:
                raise ValueError(f"Invalid observation state: {state}")

        # Step 1. Initialize variables
        n_obs = len(input_observation_states)
        prob_mat = np.zeros((self.n_states, n_obs))
       
        # Step 2. Calculate probabilities
        first_obs_idx = self.observation_states_dict[input_observation_states[0]]
        prob_mat[:, 0] = self.prior_p * self.emission_p[:, first_obs_idx]

        for t in range(1, n_obs):
            obs_idx = self.observation_states_dict[input_observation_states[t]]
            for s in range(self.n_states):
                prob_mat[s, t] = np.sum(prob_mat[:, t - 1] * self.transition_p[:, s] * self.emission_p[s, obs_idx])

        # Step 3. Return final probability
        log_prob_mat = np.log(prob_mat + 1e-300)  # Add small value to prevent log(0)
        forward_log_probability = np.logaddexp.reduce(log_prob_mat[:, -1])  # summation of log probabilities
        return np.exp(forward_log_probability)  # Convert back from log probability

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """
        decode_observation_states = np.atleast_1d(decode_observation_states)

        # edge cases
        if decode_observation_states.size == 0:
            raise ValueError("Input sequence is empty.")

        for state in decode_observation_states:
            if state not in self.observation_states_dict:
                raise ValueError(f"Invalid observation state: {state}")

        # Step 1. Initialize variables
        n_obs = len(decode_observation_states)

        # store probabilities of hidden state at each step
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))

        # store best path for traceback
        best_path = np.zeros((n_obs, self.n_states), dtype=int)

       # Step 2. Calculate Probabilities
        first_obs_idx = self.observation_states_dict[decode_observation_states[0]]
        viterbi_table[0, :] = self.prior_p * self.emission_p[:, first_obs_idx]

        for t in range(1, n_obs):
            obs_idx = self.observation_states_dict[decode_observation_states[t]]
            for s in range(self.n_states):
                trans_probs = viterbi_table[t - 1, :] * self.transition_p[:, s]

                # Handle states that have no valid transitions
                if np.all(self.transition_p[:, s] == 0):
                    raise ValueError(f"State {self.hidden_states[s]} cannot transition to any other state.")

                # Handle cases where there is no valid transition for the current state
                if np.all(trans_probs == 0):  # No valid transition exists from previous states
                    viterbi_table[t, s] = 0  # Set probability to 0 as no valid path exists
                    best_path[t, s] = 0  # Keep track of invalid path

                else:
                    best_prev_state = np.argmax(trans_probs)
                    viterbi_table[t, s] = trans_probs[best_prev_state] * self.emission_p[s, obs_idx]
                    best_path[t, s] = best_prev_state

        # Step 3. Traceback
        best_path_pointer = np.argmax(viterbi_table[-1, :])
        best_hidden_state_sequence = [self.hidden_states[best_path_pointer]]

        for t in range(n_obs - 1, 0, -1):
            best_path_pointer = best_path[t, best_path_pointer]
            best_hidden_state_sequence.insert(0, self.hidden_states[best_path_pointer])

        # Step 4. Return best hidden state sequence 
        return best_hidden_state_sequence
