import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    Create an instance of your HMM class using the "mini_weather_hmm.npz" file.
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')

    test = HiddenMarkovModel(
        mini_hmm['observation_states'],
        mini_hmm['hidden_states'],
        mini_hmm['prior_p'],
        mini_hmm['transition_p'],
        mini_hmm['emission_p']
    )

    input_observation_sequence = mini_input['observation_state_sequence']
    best_hidden_state_sequence = mini_input['best_hidden_state_sequence']

    # for Forward algorithm
    forward_prob = test.forward(input_observation_sequence)

    # Check that Forward probability is correct
    expected_forward_prob = 0.03506  # manually calculated
    assert round(forward_prob, 5) == expected_forward_prob, f"Expected forward probability: {expected_forward_prob}, but got: {round(forward_prob, 5)}"

    # Check Forward returns float and greater than zero
    assert isinstance(forward_prob, float), "Forward algorithm should return a float probability."
    assert forward_prob > 0, "Forward probability should be greater than zero."
    assert forward_prob <= 1, "Forward probability should not exceed 1."

    # for Viterbi algorithm
    viterbi_path = test.viterbi(input_observation_sequence)

    # Check that the Viterbi output matches the expected hidden state sequence
    assert list(best_hidden_state_sequence) == viterbi_path, f"Viterbi path does not match expected sequence. Expected: {list(best_hidden_state_sequence)}, but got: {viterbi_path}"

    # Check that Viterbi returns a list and length matches input sequence
    assert isinstance(viterbi_path, list), "Viterbi algorithm should return a list."
    assert len(viterbi_path) == len(
        input_observation_sequence), f"Viterbi path length should match observation length. Expected: {len(input_observation_sequence)}, but got: {len(viterbi_path)}"

    # Edge cases
    with pytest.raises(ValueError):
        test.forward(np.array([]))  # Empty input sequence
    with pytest.raises(ValueError):
        test.viterbi(np.array(['invalid_state']))  # Invalid observation state

    mini_input.close()  # Close npz file


def test_full_weather():

    """
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    test = HiddenMarkovModel(
        full_hmm['observation_states'],
        full_hmm['hidden_states'],
        full_hmm['prior_p'],
        full_hmm['transition_p'],
        full_hmm['emission_p']
    )

    input_observation_sequence = full_input['observation_state_sequence']
    best_hidden_state_sequence = full_input['best_hidden_state_sequence']

    # for Forward algorithm
    forward_prob = test.forward(input_observation_sequence)
    assert isinstance(forward_prob, float), "Forward algorithm should return a float probability."
    assert forward_prob > 0, "Forward probability should be greater than zero."
    assert forward_prob <= 1, "Forward probability should not exceed 1."

    # Run Viterbi and check output
    viterbi_path = test.viterbi(input_observation_sequence)

    # Assert that the Viterbi path matches the best hidden state sequence
    assert list(best_hidden_state_sequence) == viterbi_path, "Viterbi path doesn't match the expected state sequence."

    # Ensure that the Viterbi path is a list and its length matches the observation sequence length
    assert isinstance(viterbi_path, list), "Viterbi algorithm should return a list."
    assert len(viterbi_path) == len(input_observation_sequence), "Viterbi path should match observation length."

    full_input.close()  # Close npz file
