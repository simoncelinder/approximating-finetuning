# %load_ext autoreload
# %autoreload 2

# +
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

# +
text_example = "Once upon a time, there was a little prince, living in a beautiful castle with his step father. As it came to pass, "
tokens = tokenizer(text_example)['input_ids']
n_new_tokens = 100

print(f'Start: {tokenizer.decode(tokens)}') 
for i in range(n_new_tokens):
    res_dict = h.query_api_for_models(
        model_shortname_engine_dict=model_shortname_engine_dict,
        n_logprobs=100,
        prompt_tokens=tokens,
        tokenizer=tokenizer,
    )

    lps_list, other_empty = h.alignment_pipeline([res_dict])
    
    blend_res = h.blend_pipeline(
            lp_dicts = lps_list,
            **params,
    )

    next_token = h.sample_token(blend_res[0]['blended'])
    tokens.append(next_token)
    print(tokenizer.decode(next_token))
    
print(f"With completion: {tokenizer.decode(tokens)}")
# -




