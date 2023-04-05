# ### This notebook was used to decide between two different data structures for keeping the logprobs of the models. Since we need to align the logprobs across models, 2 approaches was times for comparison (GPT4 wrote the code).

# +
# Alternative 1) Keep one array with the logprobs and one with the token ids per model in each dict like this:
{
    f'{model_name}_logprobs': np.array(...),  # Floats with logprobs
    f'{model_name}_token_ids': np.array(...),  # Ints that describe which token the logprobs refer to
    ...
}

# Alternative 2) Nested dict where each key is the token id, and each value is the logprob of that token:
{
     f'{model_name}_logprobs': {374: -0.47, 17: -0.12, ...
}

# +
import numpy as np

# Generate random keys (token IDs) and values (logprobs) for two dicts
np.random.seed(42)
keys_1 = np.random.choice(np.arange(1, 101), 80, replace=False)
values_1 = np.random.uniform(-10, 0, 80)

keys_2 = np.random.choice(np.arange(1, 101), 60, replace=False)
values_2 = np.random.uniform(-10, 0, 60)

# Alternative 1: Separate arrays
data_array = {
    "keys_1": keys_1,
    "values_1": values_1,
    "keys_2": keys_2,
    "values_2": values_2,
}

# Alternative 2: Nested dict
data_dict = {
    "dict_1": dict(zip(keys_1, values_1)),
    "dict_2": dict(zip(keys_2, values_2)),
}

# Define functions to align values based on keys for both alternatives
def align_values_array(data: dict) -> np.ndarray:
    common_keys = np.intersect1d(data["keys_1"], data["keys_2"])
    aligned_values_1 = np.array([data["values_1"][np.where(data["keys_1"] == key)[0][0]] for key in common_keys])
    aligned_values_2 = np.array([data["values_2"][np.where(data["keys_2"] == key)[0][0]] for key in common_keys])
    return aligned_values_1, aligned_values_2

def align_values_dict(data: dict) -> np.ndarray:
    common_keys = set(data["dict_1"].keys()) & set(data["dict_2"].keys())
    aligned_values_1 = np.array([data["dict_1"][key] for key in common_keys])
    aligned_values_2 = np.array([data["dict_2"][key] for key in common_keys])
    return aligned_values_1, aligned_values_2
# -

# %timeit -r 10 -n 1000 align_values_array(data_array)

# %timeit -r 10 -n 1000 align_values_dict(data_dict)


