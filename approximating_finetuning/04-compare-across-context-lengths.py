# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import re
import pickle

import pandas as pd
import cufflinks as cf

from approximating_finetuning import helpers as h

cf.go_offline()

folder = Path('data/multiple_context_lengths')
files = list(folder.glob('*.pkl'))

params = {
    'small_untuned_temperature': 1.7655767219933314,
    'small_tuned_temperature': 0.6302178045309944,
    'big_temperature': 1.6737429561557058,
    'diff_weight': 0.1280484139350893
}



lps_paths_list = sorted([i for i in files if 'lps_list' in i.as_posix()])
other_paths_list = sorted([i for i in files if 'other_list' in i.as_posix()])
assert len(lps_paths_list) == len(other_paths_list)

res_list = []
for lps_list_path, other_list_path in zip(lps_paths_list, other_paths_list):
    tokens_back = int(re.search(r'(\d+)tb', lps_list_path.name).group(1))
    assert tokens_back == int(re.search(r'(\d+)tb', other_list_path.name).group(1))
    
    with open(lps_list_path, 'rb') as f:
        lps_list = pickle.load(f)
    with open(other_list_path, 'rb') as f:
        other_list = pickle.load(f)
        
    # Override to be like test set part of tuning notebook not to leak from val set tuning
    share_val = 0.5  # TODO make global variable for this notebook and the tuning
    n_val = int(len(lps_list)*share_val)
    lps_list = lps_list[n_val::]
    other_list = other_list[n_val::]
        
    true_tokens = [r['token'] for r in other_list]
    blended_lps_list = h.blend_pipeline(
        lps_list = lps_list,
        **params,
    )
    lps_incl_blended = h.merge_blended_and_original_lps(lps_list, blended_lps_list)
    res_dict = h.calculate_perplexity_per_model(lps_incl_blended, true_tokens, verbosity=0)
    res_dict['tokens_back'] = tokens_back
    res_list.append(res_dict)
# -

(
    pd.DataFrame(res_list)
    .set_index('tokens_back')
    .sort_index()
).iplot(dimensions=(800, 600), xTitle='Number of tokens in context', yTitle='Perplexity')


