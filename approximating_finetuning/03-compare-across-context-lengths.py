# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import re
import pickle

import pandas as pd
import cufflinks as cf
import plotly.colors as plt_colors

from approximating_finetuning import helpers as h
from approximating_finetuning.global_params import SHARE_VAL

cf.go_offline()

# +
#help(plotly.colors)

# +
# Note: We use this path for the raw API results also for the later overriding 
# with davinci base model, which is among other models in these results
folder = Path(f'data/text_davinci_003')

# Plot configs
plot_format = {
    'dimensions': (800, 500),
    'mode': 'lines+markers',
    'size': 4,
    'xTitle': 'Number of tokens in context',
    'yTitle': 'Perplexity',
    'colors': dict(
        zip(
            [
                'small_untuned',
                'small_tuned',
                'text_davinci_003',
                'text_davinci_003_blended',
                'davinci_base',
                'davinci_base_blended'], 
            plt_colors.qualitative.Plotly
        )
    )
}
# -

# ## text_davinci_003

# +
model_name = 'text_davinci_003'

params = {
    'small_untuned_temperature': 8.826880965812197,
    'small_tuned_temperature': 2.233992035896991,
    'big_temperature': 1.6695405309995934
}

# Note: The small models CAN have different results when switching big model since they are 
# aligning to what logprobs exist in the big model 
big_model_df_dict = {}

path_list = list(folder.glob('*.pkl'))
path_list = sorted([i for i in path_list if 'raw_api' in i.as_posix()])

res_list = []

for path in path_list:
    tokens_back = int(re.search(r'(\d+)tb', path.name).group(1))

    with open(path, 'rb') as f:
        res = pickle.load(f)

    # Override to be like test set part of tuning notebook not to leak from val set tuning
    n_val = int(len(res)*SHARE_VAL) # To be consistent with Optuna hyperparam tuning not to leak
    res = res[n_val::]
    
    # Just delete other base model for reducing confusion
    for r in res:
        del r['davinci_base_logprobs']

    true_tokens = [r['token'] for r in res]
    lps_list = [{k: v for k, v in r.items() if 'logprobs' in k} for r in res]

    blended_lps_list = h.blend_pipeline(
        res = res,
        **params,
    )

    lps_incl_blended = h.merge_blended_and_original_lps(lps_list, blended_lps_list)
    res_dict = h.calculate_perplexity_per_model(lps_incl_blended, true_tokens)
    res_dict['tokens_back'] = tokens_back
    res_list.append(res_dict)

davinci_3_df = (
    pd.DataFrame(res_list)
    .rename(
        columns={
            'big': f'{model_name}',
            'blended': f'{model_name}_blended'
        }
    )
    .set_index('tokens_back')
    .sort_index()
    .iloc[1::]  # Very few tokens in context skew plot
)

davinci_3_df.iplot(
    **plot_format
)
# -

# ## Override with davinci base

# +
# Note: Should not assign new folder variable here since we are 
# fetching davinci_base results from the raw api results of the initial 
# big model 
model_name = 'davinci_base'

override_params = {
    'small_untuned_temperature': 7.71104729547284,
    'small_tuned_temperature': 3.5682361618152374,
    'big_temperature': 1.1444793391335604
}

# Note: The small models CAN have different results when switching big model since they are 
# aligning to what logprobs exist in the big model 
big_model_df_dict = {}

path_list = list(folder.glob('*.pkl'))
path_list = sorted([i for i in path_list if 'raw_api' in i.as_posix()])

res_list = []

for path in path_list:
    tokens_back = int(re.search(r'(\d+)tb', path.name).group(1))

    with open(path, 'rb') as f:
        res = pickle.load(f)

    # Override to be like test set part of tuning notebook not to leak from val set tuning
    n_val = int(len(res)*SHARE_VAL) # To be consistent with Optuna hyperparam tuning not to leak
    res_val = res[n_val::]
    
    # Override big model with davinci base
    res_val_override = h.override_big_model(res_val, other_big_model_key='davinci_base_logprobs')

    true_tokens = [r['token'] for r in res_val_override]
    lps_list = [{k: v for k, v in r.items() if 'logprobs' in k} for r in res_val_override]

    blended_lps_list = h.blend_pipeline(
        res = res_val_override,
        **override_params,
    )

    lps_incl_blended = h.merge_blended_and_original_lps(lps_list, blended_lps_list)
    res_dict = h.calculate_perplexity_per_model(lps_incl_blended, true_tokens)
    res_dict['tokens_back'] = tokens_back
    res_list.append(res_dict)

davinci_base_df = (
    pd.DataFrame(res_list)
    .rename(
        columns={
            'blended': f'{model_name}_blended'
        }
    )
    .drop(columns=['big'])  # Drop instead of rename, else 2 of the same
    .set_index('tokens_back')
    .sort_index()
    .iloc[1::]  # Very few tokens in context skew plot
)

davinci_base_df.iplot(
    title= f'Results and blending with big model = {model_name}',
    **plot_format
)
# -

compare = (
    davinci_3_df
    .join(
        davinci_base_df
        [['davinci_base', 'davinci_base_blended']]
    )
)

# ## Plots for report

# +
#help(cufflinks.pd.DataFrame.iplot)
# -

compare.iplot(
    title='Full view of all 6 models including blended',
    **plot_format
)

(
    compare
    .loc[compare.index <= 25]
    [[i for i in compare.columns if i != 'small_untuned']]
    .iplot(
        #title='Small tuned model is best at very short context',
        title='Short context comparison (context length <= 25)',
        **plot_format
    )
)

(
    compare
    .loc[
        (compare.index >= 25) & 
        (compare.index <= 200)
    ]
    [[i for i in compare.columns if i not in ['small_untuned', 'small_tuned']]]
    .iplot(
        title='Long context comparison (25 <= context length <= 200)',
        **plot_format
    )
)

(
    compare
    .loc[
        (compare.index >= 25) & 
        (compare.index <= 200)
    ]
    [['text_davinci_003_blended', 'davinci_base_blended']]
    .iplot(
        title='Long context comparison of only the blended models (25 <= context length <= 200)',
        **plot_format,
    )
)


