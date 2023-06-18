# %load_ext autoreload
# %autoreload 2

# +
from IPython.display import clear_output
from copy import deepcopy

from transformers import GPT2TokenizerFast

import approximating_finetuning.helpers as h

# Define tuned params
params = {
    'small_untuned_temperature': 8.66226163868641,
    'small_tuned_temperature': 2.209180197239623,
    'big_temperature': 1.658980070394385
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
        # List wrapping for required format
        res = [
                h.query_api_for_models(
                model_shortname_engine_dict=model_shortname_engine_dict,
                n_logprobs=100,
                prompt_tokens=tokens,
                tokenizer=tokenizer,
            )
        ]

        blend_res = h.blend_pipeline(
                res = res,
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



# ## As multiplier on small model temperatures goes from inf to 1, we should see text become more like childrens books

# ### Note that this is the text-davinci-003 model blending not davinci base

# +
#text_example = "As the prince left the castle, "
#text_example = "And then he went to his father, "
text_example = "As the sun dipped below the horizon, casting a golden glow across the city, "
#text_example = "When asked about the increasingly "
n_new_tokens = 50

r = {}
#for m in [20, 4, 1]:
for m in [20]:
    
    tokens = tokenizer(text_example)['input_ids']
    words = tokenizer.decode(tokens)
    
    # Apply multiplier on the small model temperatures
    temp_params = deepcopy(params)
    temp_params['small_untuned_temperature'] *= m
    temp_params['small_tuned_temperature'] *= m
    
    try:
        for i in range(n_new_tokens):
            try:
                res = [
                        h.query_api_for_models(
                        model_shortname_engine_dict=model_shortname_engine_dict,
                        n_logprobs=100,
                        prompt_tokens=tokens,
                        tokenizer=tokenizer,
                    )
                ]

                blend_res = h.blend_pipeline(
                        res = res,
                        **temp_params,
                )

                next_token = h.sample_token(blend_res[0]['blended'])
                tokens.append(next_token)
                words += tokenizer.decode(next_token)
                words = words.replace("\n\n", "\n")
                print(words, end=" ")
                clear_output(wait=True)
            except:
                pass
        r[m] = words
    except KeyboardInterrupt:
        pass

r
# -


# r


