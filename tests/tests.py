import numpy as np

import approximating_finetuning.helpers as h


def test_normalize_logprobs():
    n_decimals = 5
    mock_big_probs = np.array([[0.1, 0.2], [0.2, 0.2]])
    mock_blended_probs = np.array([[0.7, 0.1], [0.2, 0.8]])
    mock_big_lps = np.log(mock_big_probs)
    mock_blended_lps = np.log(mock_blended_probs)
    res_lps = h.normalize_logprobs(mock_blended_lps, mock_big_lps)
    res_probs = np.exp(res_lps)
    assert (np.sum(res_probs, axis=1).round(n_decimals) == np.sum(mock_big_probs, axis=1).round(n_decimals)).all()


def test_scale_logprobs():

    # TODO: Written with GPT4 - Might want to double check that the concept is correct
    logprobs_dict = {1: -0.5, 2: -1.0, 3: -1.5, 4: -2.0}
    temperature = 2.0

    expected_scaled_logprobs = {1: -0.25, 2: -0.5, 3: -0.75, 4: -1.0}
    actual_scaled_logprobs = h.scale_logprobs_with_temperature(logprobs_dict, temperature)

    # Test that the keys are the same
    assert set(expected_scaled_logprobs.keys()) == set(actual_scaled_logprobs.keys()), "Keys do not match."

    # Test that the values are almost equal (within a tolerance)
    for key in expected_scaled_logprobs.keys():
        assert np.isclose(
            expected_scaled_logprobs[key], actual_scaled_logprobs[key], atol=1e-8
        ), f"Values for key {key} do not match: {expected_scaled_logprobs[key]} != {actual_scaled_logprobs[key]}"
