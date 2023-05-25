# +
# %load_ext autoreload
# %autoreload 2

import pickle
import joblib
import datetime

import optuna

import approximating_finetuning.helpers as h
from approximating_finetuning.global_params import SHARE_VAL

# +
# Load data prepared and aligned by the prep notebook
look_for_n_rows = 495

with open(f'data/500_ex/lps_list_{look_for_n_rows}.pkl', 'rb') as f:
    lps_list = pickle.load(f)
    
with open(f'data/500_ex/other_list_{look_for_n_rows}.pkl', 'rb') as f:
    other_list = pickle.load(f)

# +
# We previously distinguished only train and test to ensure not fine tuning
# a small base model through OpenAI API on test data.
# We now also split the test set into val (for tuning) and test (final score), 
# to ensure hyperparameter tuning isnt overfitted.
# We use the imported global variable for consistency across to other notebooks
share_val = SHARE_VAL  
n_val = int(len(lps_list)*share_val)

lps_list_val = lps_list[0:n_val]
other_list_val = other_list[0:n_val]

lps_list_test = lps_list[n_val::]
other_list_test = other_list[n_val::]

true_tokens_val = [r['token'] for r in other_list_val]
true_tokens_test = [r['token'] for r in other_list_test]
# -

# Baselines
# TODO: Presumably the 4.85% nan is just when the true token didnt exist even in davinci?
h.calculate_perplexity_per_model(lps_list_test, true_tokens_test)

# +
study_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
study = optuna.create_study()
n_trials = 100
    
def objective(trial):
    
    blended_lps_list_val = h.blend_pipeline(
        lps_list = lps_list_val,
        small_untuned_temperature=trial.suggest_float('small_untuned_temperature', 0.5, 2),
        small_tuned_temperature=trial.suggest_float('small_tuned_temperature', 0.5, 2),
        big_temperature=trial.suggest_float('big_temperature', 0.5, 2),
        diff_weight=trial.suggest_float('diff_weight', 0.05, 1.0),
    )
        
    # Since we want to be consistent in matching to the existing tokens of the big model
    # Even when just calculating ppl for the blended:
    lps_incl_blended_val = h.merge_blended_and_original_lps(lps_list, blended_lps_list_val)
    res_dict = h.calculate_perplexity_per_model(lps_incl_blended_val, true_tokens_val, verbosity=0)
    ppl = res_dict['blended']
    return ppl

study.optimize(objective, n_trials=n_trials)
# -

study.best_params

# +
blended_lps_list_test = h.blend_pipeline(
    lps_list = lps_list_test,
    **study.best_params,   
)

lps_incl_blended_test = h.merge_blended_and_original_lps(lps_list_test, blended_lps_list_test)
res_dict = h.calculate_perplexity_per_model(lps_incl_blended_test, true_tokens_test, verbosity=0)
res_dict

# +
#joblib.dump(study, f'hyperparam_studies/{study_name}.pkl')
# -


