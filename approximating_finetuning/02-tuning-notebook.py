# +
# %load_ext autoreload
# %autoreload 2

import pickle
import joblib
import datetime

import optuna

import approximating_finetuning.helpers as h

# +
# Load data prepared and aligned by the prep notebook
look_for_n_rows = 495

with open(f'data/lps_list_{look_for_n_rows}.pkl', 'rb') as f:
    lps_list = pickle.load(f)
    
with open(f'data/other_list_{look_for_n_rows}.pkl', 'rb') as f:
    other_list = pickle.load(f)
# -

true_tokens = [r['token'] for r in other_list]

# Baselines
# TODO: Presumably the 4.85% nan is just when the true token didnt exist even in davinci?
h.calculate_perplexity_per_model(lps_list, true_tokens)

# +
study_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
study = optuna.create_study()
n_trials = 100
    
def objective(trial):
    blend_res = h.blend_pipeline(
        lp_dicts = lps_list,
        small_untuned_temperature=trial.suggest_float('small_untuned_temperature', 0.5, 2),
        small_tuned_temperature=trial.suggest_float('small_tuned_temperature', 0.5, 2),
        big_temperature=trial.suggest_float('big_temperature', 0.5, 2),
        diff_weight=trial.suggest_float('diff_weight', 0.05, 1.0),
    )
    
    # TODO simplify
    ppl_all = h.calculate_perplexity_per_model(blend_res, true_tokens)
    ppl = ppl_all['blended']
    return ppl

study.optimize(objective, n_trials=n_trials)
# -

study.best_params

{
    **{'blended': study.best_value},
    **h.calculate_perplexity_per_model(lps_list, true_tokens)
}

joblib.dump(study, f'hyperparam_studies/{study_name}.pkl')




