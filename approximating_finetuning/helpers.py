import os
import random
from typing import List, Dict, Any, Callable, Union

import numpy as np
import pandas as pd
import openai
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast


api_key_abspath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_key')


with open(api_key_abspath, 'r') as f:
    try:
        openai.api_key = f.read()
    except Exception as e:
        print("Seems you have not provided your api key in file! Error:", e)


def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def sample_context_and_completion(
    text: str,
    n_words_context: int = 150,
    n_words_completion: int = 100,
    drop_partial_start: bool = True,
    drop_partial_end: bool = True,
) -> tuple:
    words = text.split(' ')
    min_ = n_words_context
    max_ = len(words) - n_words_completion
    breakpoint = random.randint(min_, max_)
    context = ' '.join(words[breakpoint - n_words_context: breakpoint])
    completion = ' '.join(words[breakpoint: breakpoint + n_words_completion])
    if drop_partial_start:
        context = ' '.join(context.split('.')[1:])
    if drop_partial_end:
        trunc_end = '.'.join(completion.split('.')[0:-1]) + '.'
        if len(trunc_end) > 0 and not completion[-1] == '.':
            completion = trunc_end
    return context, completion


def evenly_distributed_sampling(text: str, n_samples: int = 5, sample_n_words: int = None) -> list:
    words = text.split(' ')
    samples = []
    step_size = len(words) // n_samples
    if sample_n_words is None:
        sample_n_words = step_size
        print("Sample size (words per sample) not specified, derived from step size:", step_size)
    for i in range(0, len(words) - step_size, step_size):
        samples.append(' '.join(words[i:i + sample_n_words]))
    return samples


def generate_text(
    prompt: str,
    engine: str = 'babbage:ft-personal-2022-07-06-01-24-58',
    temperature: float = None,
    max_tokens: int = None,
) -> str:
    raw_response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return raw_response['choices'][0]['text']


def generate_logprobs(
    prompt: Union[str, list],
    engine: str = 'babbage:ft-personal-2022-07-06-01-24-58',
    temperature: float = None,
    n_logprobs: int = 10,
) -> Dict[str, float]:  # {'some_word': -0.12, ...}}

    r = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=1,  # Only support predicting 1 token ahead in this function
        logprobs=n_logprobs,
    )

    # Has text as key, not token id
    return r['choices'][0].__dict__['_previous']['logprobs']['top_logprobs'][0]


def tokenize_keys_in_dict(
    dict_: Dict[str, float],
    tokenizer: GPT2TokenizerFast,
) -> Dict[int, float]:
    tokenized_dict = {}
    for token, val in dict_.items():
        token_int = tokenizer(token)['input_ids'][0]
        tokenized_dict[token_int] = val
    return tokenized_dict


def lookup_proba_true_token(true_token: int, model_prob_dict: dict, proba_when_not_found: float = -np.inf) -> float:
    if true_token in model_prob_dict.keys():
        res_for_token = model_prob_dict[true_token]
    else:
        res_for_token = proba_when_not_found
    return res_for_token


def calc_perplexity(proba_series: pd.Series) -> float:
    # Aligns with: https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3

    # Normalize with number of predicted tokens
    N = len(proba_series)

    # P(w_1, w_2, ... w_n) = p(w_1) * p(w_2) * ... * p(w_n)
    combined_proba = proba_series.prod()

    # Pexplexity = Nth root of 1/P
    return (1 / combined_proba) ** (1 / N)


def calc_perplexity_chunkwise(s: pd.Series, rows_per_chunk: int = 50, n_decimals: int = 2) -> float:
    # This case required for when passing fewer rows than chunk size, else perplexity can become 0
    if len(s) <= rows_per_chunk:
        return calc_perplexity(s)

    # Due to zero-division when chaining too many probabilities, we chunk the series
    # and average the perplexity of the chunks
    return round(
        np.mean(
            [
                calc_perplexity(s.iloc[i:i + rows_per_chunk])
                for i in range(0, len(s), rows_per_chunk)
            ]
        ), n_decimals
    )


def calculate_perplexity_per_model(
    result_list: List[Dict[str, np.ndarray]],
    true_tokens: List[str],
    n_decimals: int = 1,
) -> Dict[str, float]:
    true_probs_per_model = []
    for i, true_token in enumerate(true_tokens):
        d = result_list[i]
        if true_token in d['big_logprobs'].keys():
            true_probs_per_model.append(({k: v[true_token] for k, v in d.items()}))
        else:
            true_probs_per_model.append(({k: np.nan for k in d.keys()}))

    true_lps_df = pd.DataFrame(true_probs_per_model)
    if true_lps_df.isnull().values.any():
        # Seems like same num nans across models and stays constant during tuning, so probably not "gaming" the nans
        percent_nan = true_lps_df.isnull().sum().sum() / (true_lps_df.shape[0] * true_lps_df.shape[1]) * 100
        is_same_rows_that_are_nan = (true_lps_df.isna().all(axis=1) == true_lps_df.isna().any(axis=1)).all()
        print(f"{is_same_rows_that_are_nan=}")
        print(
            f"WARNING: {percent_nan :.2f}% NaNs in true_lps_df for calculating perplexity - They will be dropped. "
            f"Exact same nan across models = {is_same_rows_that_are_nan}."
        )
        true_lps_df = true_lps_df.dropna()

    ppl_per_model = {}
    for c in true_lps_df.columns:
        probs = np.exp(true_lps_df[c])
        # TODO check why n_decimals doesnt work when passed into calc_perplexity_chunkwise
        ppl_per_model[c] = round(calc_perplexity_chunkwise(probs), n_decimals)

    return ppl_per_model


def calc_perplexity_copilot(probabilities: pd.Series) -> float:
    # As autocompleted by Copilot, gives same result as function above but more compact
    return np.exp(-np.sum(np.log(probabilities)) / len(probabilities))


def model_results_to_list_of_dicts(
    tokens,
    small_untuned_model: str = 'text-babbage-001',
    small_tuned_model: str = 'babbage:ft-cyborgs-2023-02-02-11-09-21',
    big_model: str = 'text-davinci-003',
    tokens_back: int = 300,
    max_iter: int = 5,
    n_logprobs: int = 5,
    step_size: int = None,
    tokenizer: GPT2TokenizerFast = None,
) -> List[dict]:
    print("Warning! This function queries the openai api in a loop")

    # Best to use general names for models, so helper functions dont need to be
    # aware of exact models queried
    model_shortname_engine_dict = {
        'small_untuned': small_untuned_model,
        'small_tuned': small_tuned_model,
        'big': big_model,
    }

    if step_size is None:
        # Sample evenly across test set
        step_size = len(tokens) // max_iter

    if tokenizer is None:
        print("No tokenizer provided, using gpt2 tokenizer. ")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    result_list = []
    count = 1
    for i in range(tokens_back, len(tokens), step_size):
        if count > max_iter:
            break
        if i % 5 == 0:
            print("Iteration: ", count)
        try:

            # Increment rolling window
            prompt_tokens = tokens[i: i + tokens_back]
            prompt_text = tokenizer.decode(prompt_tokens)
            true_token = tokens[i + tokens_back]
            true_word = tokenizer.decode([true_token])

            res_dict = query_api_for_models(
                model_shortname_engine_dict=model_shortname_engine_dict,
                n_logprobs=n_logprobs,
                prompt_tokens=prompt_tokens,
                tokenizer=tokenizer,
            )


            res_dict = assign_shared_metadata(prompt_text, res_dict, true_token, true_word)

            # Only append after all models have been queried
            result_list.append(res_dict)

        except Exception as e:
            print(f"Iterration {i} failed, Error message: {e}")
        count += 1

    return result_list


def query_api_for_models(
    model_shortname_engine_dict: dict,
    n_logprobs: int,
    prompt_tokens: list,
    tokenizer: GPT2TokenizerFast,
):
    res_dict = {}
    for model_shortname, engine in model_shortname_engine_dict.items():
        lp_dict_text_keys = generate_logprobs(
            engine=engine,
            prompt=prompt_tokens,
            n_logprobs=n_logprobs
        )
        lp_dict_token_keys = tokenize_keys_in_dict(lp_dict_text_keys, tokenizer=tokenizer)
        res_dict[f'{model_shortname}_logprobs'] = lp_dict_token_keys
    return res_dict


def assign_shared_metadata(
        prompt_text: str,
        res_dict: dict,
        true_token: int,  # TODO int or string?
        true_word: str,
) -> dict:
    res_dict['token'] = true_token
    res_dict['word'] = true_word
    res_dict['prompt_text'] = prompt_text
    return res_dict


def train_test_split(text: str, share_test: float) -> tuple:
    split_idx = int(len(text) * (1 - share_test))
    train = text[0:split_idx]
    test = text[split_idx:]
    return train, test


def scale_logprobs_with_temperature(logprobs_dict: dict, temperature: float) -> dict:
    # Convert the logprobs dictionary to a NumPy array
    keys, values = zip(*logprobs_dict.items())
    logprobs_array = np.array(values)

    # Apply the temperature scaling
    scaled_logprobs_array = logprobs_array / temperature

    # Convert the scaled logprobs back to a dictionary
    scaled_logprobs_dict = dict(zip(keys, scaled_logprobs_array))
    return scaled_logprobs_dict


def scale_logprobs_for_the_3_models(
    small_untuned_logprobs: Dict[int, float],
    small_tuned_logprobs: Dict[int, float],
    big_logprobs: Dict[int, float],
    small_untuned_temperature: float,
    small_tuned_temperature: float,
    big_temperature: float,
) -> Dict[str, Dict[int, float]]:
    return {
        "small_untuned_logprobs": scale_logprobs_with_temperature(small_untuned_logprobs, small_untuned_temperature),
        "small_tuned_logprobs": scale_logprobs_with_temperature(small_tuned_logprobs, small_tuned_temperature),
        "big_logprobs": scale_logprobs_with_temperature(big_logprobs, big_temperature),
    }


def blend_logprobs_with_weight(
    small_untuned_logprobs: Dict[int, float],
    small_tuned_logprobs: Dict[int, float],
    big_logprobs: Dict[int, float],
    diff_weight: float,
) -> Dict[int, float]:
    # Assumes:
    # Already aligned all token ids so they are the exact same across models
    # No nan values, they have already been padded with min floats
    assert small_untuned_logprobs.keys() == small_tuned_logprobs.keys() == big_logprobs.keys()
    diffs = (np.array(list(small_tuned_logprobs.values())) - np.array(list(small_untuned_logprobs.values())))
    blended_arr = np.array(list(big_logprobs.values())) + diff_weight * diffs
    return dict(zip(big_logprobs.keys(), blended_arr))


def align_logprobs_to_big_model(
    big_logprobs: Dict[int, float],
    small_untuned_logprobs: Dict[int, float],
    small_tuned_logprobs: Dict[int, float],
) -> Dict[str, Dict[int, float]]:
    big_keys = set(big_logprobs.keys())

    # Drop any keys that are not in big model logprobs
    aligned_small_untuned = {k: v for k, v in small_untuned_logprobs.items() if k in big_keys}
    aligned_small_tuned = {k: v for k, v in small_tuned_logprobs.items() if k in big_keys}

    # Add missing keys from big_keys and set their values to np.nan
    for k in big_keys:
        if k not in aligned_small_untuned:
            aligned_small_untuned[k] = np.nan
        if k not in aligned_small_tuned:
            aligned_small_tuned[k] = np.nan

    # Sort the dictionaries by keys
    big_logprobs = dict(sorted(big_logprobs.items()))
    aligned_small_untuned = dict(sorted(aligned_small_untuned.items()))
    aligned_small_tuned = dict(sorted(aligned_small_tuned.items()))
    assert big_logprobs.keys() == aligned_small_untuned.keys() == aligned_small_tuned.keys()

    return {
        'big_logprobs': big_logprobs,
        'small_untuned_logprobs': aligned_small_untuned,
        'small_tuned_logprobs': aligned_small_tuned,
    }


def impute_nan_with_min_value(logprobs_dict: Dict[int, float]) -> Dict[int, float]:
    token_ids = np.array(list(logprobs_dict.keys()))
    logprobs = np.array(list(logprobs_dict.values()))

    min_logprob = np.nanmin(logprobs)
    logprobs[np.isnan(logprobs)] = min_logprob

    imputed_logprobs_dict = dict(zip(token_ids, logprobs))
    return imputed_logprobs_dict


def normalize_logprobs(logprobs_dict: Dict[int, float]) -> Dict[int, float]:

    # Convert the logprobs dictionary to a NumPy array
    logprobs = np.array(list(logprobs_dict.values()))
    token_ids = np.array(list(logprobs_dict.keys()))

    # Compute log normalization constant
    log_norm_constant = np.log(np.sum(np.exp(logprobs)))

    # Subtract the log normalization constant from each logprob
    normalized_logprobs = logprobs - log_norm_constant

    # Create a new dict with normalized logprobs
    normalized_logprobs_dict = dict(zip(token_ids, normalized_logprobs))

    return normalized_logprobs_dict


def apply_func_to_all_models(
    fn: Callable[[Dict[int, float]], Dict[int, float]],
    lp_dict: Dict[str, Dict[int, float]],
) -> Dict[str, Any]:
    # Agnostic so can also take 'blended' or other added model
    return {model: fn(logprobs) for model, logprobs in lp_dict.items()}


def alignment_pipeline(res: List[Dict[str, Any]]):
    lp_keys = ['small_untuned_logprobs', 'small_tuned_logprobs', 'big_logprobs']
    lp_list = []
    other_list = []  # Will just be empty during pure inference

    # Split up logprobs vs other for cleaner helpers
    for lp_dict in res:
        lp_list.append({k: v for k, v in lp_dict.items() if k in lp_keys})
        other_list.append({k: v for k, v in lp_dict.items() if k not in lp_keys})

    aligned_lps_list = []
    for lp_dict in lp_list:
        aligned_lps_list.append(align_logprobs_to_big_model(**lp_dict))

    imputed_lps_list = []
    for lp_dict in aligned_lps_list:
        imp_res = apply_func_to_all_models(impute_nan_with_min_value, lp_dict)
        imputed_lps_list.append(imp_res)

    return imputed_lps_list, other_list


def blend_pipeline(
    lp_dicts: List[Dict[str, np.ndarray]],
    small_untuned_temperature: float,
    small_tuned_temperature: float,
    big_temperature: float,
    diff_weight: float,
) -> List[Dict[str, np.ndarray]]:
    rescaled_with_temp = []
    for lp_dict in lp_dicts:
        rescaled_with_temp.append(
            scale_logprobs_for_the_3_models(
                **lp_dict,
                small_untuned_temperature=small_untuned_temperature,
                small_tuned_temperature=small_tuned_temperature,
                big_temperature=big_temperature,
            )
        )

    incl_blended = []
    for lp_dict in rescaled_with_temp:
        blended_lps = blend_logprobs_with_weight(**lp_dict, diff_weight=diff_weight)
        incl_blended.append({
            **lp_dict,
            **{'blended': blended_lps}
            }
        )

    final_incl_blend = []
    for lp_dict in incl_blended:
        final_incl_blend.append(apply_func_to_all_models(normalize_logprobs, lp_dict))

    return final_incl_blend


def sample_token(
    lp_dict: Dict[int, float]  # {token_id: logprob}
) -> int:
    # Convert log probabilities to probabilities
    probs = np.exp(np.array(list(lp_dict.values())))

    # Assert probs sum to approximately 1 (should already be normalized)
    assert np.isclose(probs.sum(), 1.0)

    # Sample a token ID from the probability distribution
    # Make sure to int convert else will get a numpy int64 that does not work for the openai api
    sampled_token_id = int(np.random.choice(list(lp_dict.keys()), p=probs))

    return sampled_token_id