import numpy as np

import approximating_finetuning.helpers as h


def test_normalize_logprobs_softmax():
    # Input dictionary
    logprobs_dict = {1: -2.0, 2: -1.0, 3: 0.0}

    # Expected output dictionary
    expected_output = {1: -2.4076059644443806, 2: -1.4076059644443806, 3: -0.4076059644443806}

    # Call the function to get the actual output
    actual_output = h.normalize_logprobs_softmax(logprobs_dict)

    # Compare the actual and expected outputs
    for k, v in expected_output.items():
        assert np.isclose(actual_output[k], v)


