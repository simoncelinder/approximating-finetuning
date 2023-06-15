# +
# %load_ext autoreload
# %autoreload 2

import re
import pickle
import joblib
import datetime
from pathlib import Path

import optuna

import approximating_finetuning.helpers as h
from approximating_finetuning.global_params import SHARE_VAL

# +
# We previously distinguished only train and test to ensure not fine tuning
# a small base model through OpenAI API on test data.
# We now also split the test set into val (for tuning) and test (final score), 
# to ensure hyperparameter tuning isnt overfitted.
# We use the imported global variable for consistency across to other notebooks
share_val = SHARE_VAL  
n_trials = 100

# Avoid overemphasizing very short contexts with very high losses
min_tokens_back_tuning = 25
# -

# Use for both text-davinci-003 and for davinci base since latter is contained in
# the raw API results
folder = Path(f'data/text_davinci_003')

# +
path_list = list(folder.glob('*.pkl'))
path_list = sorted([i for i in path_list if 'raw_api' in i.as_posix()])

# Get results for all contexts to average across them all in the blending
res_val = []
res_test = []
for path in path_list:
    tokens_back = int(re.search(r'(\d+)tb', path.name).group(1))
    if tokens_back >= min_tokens_back_tuning:
        with open(path, 'rb') as f:
            res = pickle.load(f)
            n_val = int(len(res)*share_val)
            res_val += res[0:n_val]
            res_test += res[n_val::]

# +
true_tokens_val = [r['token'] for r in res_val]
lps_list_val = [{k: v for k, v in r.items() if 'logprobs' in k} for r in res_val]

true_tokens_test = [r['token'] for r in res_test]
lps_list_test = [{k: v for k, v in r.items() if 'logprobs' in k} for r in res_test]

# +
study_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
study = optuna.create_study(study_name=study_name)
    
def tune_hypers_big_model(trial):
    blended_lps_list_val = h.blend_pipeline(
        res = res_val,
        small_untuned_temperature=trial.suggest_float('small_untuned_temperature', 0.05, 10.0),
        small_tuned_temperature=trial.suggest_float('small_tuned_temperature', 0.05, 10.0),
        big_temperature=trial.suggest_float('big_temperature', 0.05, 5.0),
    )
        
    # Since we want to be consistent in matching to the existing tokens of the big model
    # Even when just calculating ppl for the blended:
    lps_incl_blended_val = h.merge_blended_and_original_lps(lps_list_val, blended_lps_list_val)
    res_dict = h.calculate_perplexity_per_model(lps_incl_blended_val, true_tokens_val)
    ppl = res_dict['blended']
    return ppl

study.enqueue_trial({
    'small_untuned_temperature': 6.0,
    'small_tuned_temperature': 2.0,
    'big_temperature': 1.5,
})
study.optimize(tune_hypers_big_model, n_trials=n_trials)
# -

study.best_params

# +
# Example comparison of perplexity
blended_lps_list_test = h.blend_pipeline(
    res = res_test,
    **study.best_params,   
)

lps_incl_blended_test = h.merge_blended_and_original_lps(lps_list_test, blended_lps_list_test)
res_dict = h.calculate_perplexity_per_model(lps_incl_blended_test, true_tokens_test)
res_dict

# +
#joblib.dump(study, f'hyperparam_studies/{study_name}.pkl')
# -
# ## Override big model with other big model (davinci base)


# +
override_study_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
override_study = optuna.create_study(study_name=override_study_name)

res_val_override = h.override_big_model(res_val, other_big_model_key='davinci_base_logprobs')
res_test_override = h.override_big_model(res_test, other_big_model_key='davinci_base_logprobs')


def tune_hypers_override_big_model(trial):
    blended_lps_list_val = h.blend_pipeline(
        res = res_val_override,
        small_untuned_temperature=trial.suggest_float('small_untuned_temperature', 0.05, 10.0),
        small_tuned_temperature=trial.suggest_float('small_tuned_temperature', 0.05, 10.0),
        big_temperature=trial.suggest_float('big_temperature', 0.05, 5.0),
    )
        
    # Since we want to be consistent in matching to the existing tokens of the big model
    # Even when just calculating ppl for the blended:
    lps_incl_blended_val = h.merge_blended_and_original_lps(lps_list_val, blended_lps_list_val)
    res_dict = h.calculate_perplexity_per_model(lps_incl_blended_val, true_tokens_val)
    ppl = res_dict['blended']
    return ppl

override_study.enqueue_trial({
    'small_untuned_temperature': 6.0,
    'small_tuned_temperature': 2.0,
    'big_temperature': 1.5,
})
override_study.optimize(tune_hypers_override_big_model, n_trials=n_trials)
# -

override_study.best_params

# +
# Example comparison of perplexity
override_blended_lps_list_test = h.blend_pipeline(
    res = res_test,
    **override_study.best_params,
)

override_lps_incl_blended_test = h.merge_blended_and_original_lps(lps_list_test, override_blended_lps_list_test)
override_res_dict = h.calculate_perplexity_per_model(override_lps_incl_blended_test, true_tokens_test)
override_res_dict
# -


