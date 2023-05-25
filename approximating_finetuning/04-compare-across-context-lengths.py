# - [ ] Add like +200 examples by offsetting start and using similar stride => 500
# - [ ] Add code-davinci-002 (gpt3.5) or just "davinci" if not (not fine tuned, just gpt3)
# - [ ] Export the plot dataframe (no smoothing) to Nicholas

# +
#lps_list[0]
# -

folder.as_posix()

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import re
import pickle

import pandas as pd
import cufflinks as cf

from approximating_finetuning import helpers as h
from approximating_finetuning.global_params import SHARE_VAL

cf.go_offline()

n_iter = 20

model_folder_to_hypers = {

    Path(f'data/text_davinci_003'): {
        'small_untuned_temperature': 1.7655767219933314,
        'small_tuned_temperature': 0.6302178045309944,
        'big_temperature': 1.6737429561557058,
        'diff_weight': 0.1280484139350893
    },
    
    # TODO need to update with tuning this one instead
    Path(f'data/davinci_base'): {
        'small_untuned_temperature': 1.7655767219933314,
        'small_tuned_temperature': 0.6302178045309944,
        'big_temperature': 1.6737429561557058,
        'diff_weight': 0.1280484139350893
    },
}


# Note: The small models CAN have different results when switching big model since they are 
# aligning to what logprobs exist in the big model 
big_model_df_dict = {}
for folder, params in model_folder_to_hypers.items():
    files = list(folder.glob('*.pkl'))
    
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
        n_val = int(len(lps_list)*SHARE_VAL) # To be consistent with Optuna hyperparam tuning not to leak
        lps_list = lps_list[n_val::]
        other_list = other_list[n_val::]
        
        #print(lps_list[0]['small_tuned_logprobs'])
        true_tokens = [r['token'] for r in other_list]
        blended_lps_list = h.blend_pipeline(
            lps_list = lps_list,
            **params,
        )
        lps_incl_blended = h.merge_blended_and_original_lps(lps_list, blended_lps_list)
        res_dict = h.calculate_perplexity_per_model(lps_incl_blended, true_tokens, verbosity=0)
        res_dict['tokens_back'] = tokens_back
        res_list.append(res_dict)
        
    df = (
        pd.DataFrame(res_list)
        .set_index('tokens_back')
        .sort_index()
    )
    big_model = folder.name
    df.iplot(
        dimensions=(800, 400), 
        xTitle='Number of tokens in context', 
        yTitle=f'Perplexity', 
        title= f'Results and blending with big model = {big_model}'
    )
    big_model_df_dict[big_model] = df
    
# Compare the blends
(
    big_model_df_dict['davinci_base']['blended'].to_frame(f'davinci_base_blend')
    .join(
        big_model_df_dict['text_davinci_003']['blended'].to_frame(f'text_davinci_003_blend')
    )
).iplot(
    dimensions=(800, 400), 
    xTitle='Number of tokens in context', 
    yTitle=f'Perplexity', 
    title= f'Result comparisong of the blends using different big models'
)
# -





