# %load_ext autoreload
# %autoreload 2

# +
from IPython.display import clear_output

from transformers import GPT2TokenizerFast

import approximating_finetuning.helpers as h

# Define tuned params
params = {
    'small_untuned_temperature': 1.7713928057054265,
    'small_tuned_temperature': 0.5583126113606277,
    'big_temperature': 1.9916800757824642,
    'diff_weight': 0.2312181076944707,
}

model_shortname_engine_dict = {
    'small_untuned': 'text-babbage-001',
    'small_tuned': 'babbage:ft-cyborgs-2023-02-02-11-09-21',
    'big': 'text-davinci-003',  
}

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
# -

n_new_tokens = 20
text_example = "Once upon a time, there was a little prince, living in a beautiful castle with his step father. As it came to pass, "

# ## Blended model

# +
tokens = tokenizer(text_example)['input_ids']
words = tokenizer.decode(tokens)

try:
    for i in range(n_new_tokens):
        res_dict = h.query_api_for_models(
            model_shortname_engine_dict=model_shortname_engine_dict,
            n_logprobs=100,
            prompt_tokens=tokens,
            tokenizer=tokenizer,
        )

        lps_list, _ = h.alignment_pipeline([res_dict])

        blend_res = h.blend_pipeline(
                lps_list = lps_list,
                **params,
        )

        next_token = h.sample_token(blend_res[0]['blended'])
        tokens.append(next_token)
        words += tokenizer.decode(next_token)
        words = words.replace("\n\n", "\n")
        print(words, end=" ")

        clear_output(wait=True)
except KeyboardInterrupt:
    pass
# -
# ## Comparison model

comparison_model = 'text-davinci-003'
words_compare = words  # Immutable == Ok
try:
    for i in range(n_new_tokens):
        words_compare += h.generate_text(words_compare, comparison_model)
        words_compare = words_compare.replace("\n\n", "\n")
        print(words_compare, end=" ")
        clear_output(wait=True)
except KeyboardInterrupt:
    pass



