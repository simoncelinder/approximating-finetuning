# ## This notebook is for preparing a dataset by looping over models and querying them with different text context. 
# - You might jump directly to the second notebook and just use the already prepared parquet file from the open Google Drive
# - The notebook assumes you have already tuned a model on this dataset that you can use, or access the already tuned through the Cyborgs organization in OpenAI
#

# %load_ext autoreload
# %autoreload 2

# +
import pickle

import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast
from approximating_finetuning.global_params import SHARE_TEST

import approximating_finetuning.helpers as h

# +
# File downloaded into data folder from (and shortened its file name): 
# https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus?resource=download
text = h.read_file('data/fairy_tales.txt')

# Re-use same share_test that was used for splitting train data to
# tune small model on to avoid overfitted evaluation
train, test = h.train_test_split(text, share_test=SHARE_TEST)
# -

# %%time
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
test_tokens = tokenizer(test)['input_ids']  # Note: Using test since finetuned on train

# Set input params
#max_iter = 2000  # Original
max_iter = 3
n_logprobs = 100
tokens_back = 300

res = h.model_results_to_list_of_dicts(
    tokens=test_tokens,
    tokens_back=tokens_back,
    max_iter=max_iter,
    tokenizer=tokenizer,
    n_logprobs=n_logprobs,
)

# Dump raw before alignment if need to backtrace anything about alignment pipeline later
with open(f"data/raw_api_results_{len(res)}.pkl", "wb") as f:
    pickle.dump(res, f)

lps_list, other_list = h.alignment_pipeline(res)

# Example perplexity
true_tokens = [r['token'] for r in res]
h.calculate_perplexity_per_model(lps_list, true_tokens)

# +
# Dump to files to read into tuning notebook
with open(f"data/lps_list_{len(lps_list)}.pkl", "wb") as f:
    pickle.dump(lps_list, f)
    
with open(f"data/other_list_{len(other_list)}.pkl", "wb") as f:
    pickle.dump(other_list, f)
# -


