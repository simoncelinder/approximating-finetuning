# ## This notebook is for preparing a dataset by looping over models and querying them with different text context. 
# - You might jump directly to the second notebook and just use the already prepared parquet file from the open Google Drive
# - The notebook assumes you have already tuned a model on this dataset that you can use, or access the already tuned through the Cyborgs organization in OpenAI
#

# +
# %load_ext autoreload
# %autoreload 2

import pickle
from pathlib import Path

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

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
test_tokens = tokenizer(test)['input_ids']  # Note: Using test since finetuned on train

# Set input params
max_iter = 600
n_logprobs = 100

# +
# Same subfolder will be used also for the added big model
subfolder = 'text_davinci_003'
add_models_dict = {'davinci_base': 'davinci'}    # Regular davinci is GPT3 / base

# Could be refactored more nicely:
subfolder_path = Path(f"data/{subfolder}")
subfolder_path.mkdir(parents=True, exist_ok=True)

#Actual: [6, 10, 15, 25, 35, 50, 63, 75, 88, 100, 110, 125, 138, 150, 175, 200, 250, 275, 300]
for tokens_back in [13, 20, 30, 43, 55, 70, 95]:

    print(f'\n- - - - {tokens_back=} - - - -')
    res = h.model_results_to_list_of_dicts(
        tokens=test_tokens,
        tokens_back=tokens_back,
        max_iter=max_iter,
        tokenizer=tokenizer,
        n_logprobs=n_logprobs,
        add_models_dict=add_models_dict,
    )
    
    # Dump raw before alignment
    with open(f"data/{subfolder}/raw_api_results_{len(res)}ex_{tokens_back}tb.pkl", "wb") as f:
        pickle.dump(res, f)
        
# -






