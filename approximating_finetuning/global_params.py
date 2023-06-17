# For avoiding fine tuning small model and tuning hyperparams and evaluating on same data (overfitting)

# This first share test is used to ensure we dont fine tune the small base model (calling OpenAIs fine tuning API)
# so we will only do continued analysis on this test set
SHARE_TEST = 0.4

# Later in the analysis we devide up the test set into a validation set for hyperparameter tuning with Optuna,
# tuning the weight on the diffs and the temperatures, and a final test set for evaluating the final model
SHARE_VAL = 0.5