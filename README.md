# Approximating Finetuning

## Intro

We might not have access or time to directly finetune state of the art models. It is pretty accessible to finetune small models, however, and so the question is: Can we get some of the benefits from finetuning by using a small finetuned model to "steer" a larger model's generations?

Roughly my plan is to build a system composed of 3 models:
1) A base Babbage model (no finetuning)
2) A finetuned Babbage model (Jan Kirchner has a good model we could borrow for a proof of concept, but we could also make our own)
3) Base GPT-3 (no finetuning)

Next we find some way to gently nudge the generations of GPT-3 by measuring the difference between the two Babbage models, and applying it directly to the GPT-3 logits (unsure exactly how, and this might require some empirical exploration). We could also maybe do some kind of local search using the Babbage models (since they are really cheap to run), and use the results of that search to inform the GPT-3 generation. 

The goal is to build a composite system which is more useful to generate from than any of the 3 models on their own.

More details [here](https://docs.google.com/document/d/1hkz5-mvJB5LIKtFHvsbe9dq1ThchVrITSak3phN4LI8/edit?usp=sharing).


## Create virtualenv, activate it and install dependencies
First make sure you are in the root folder of this repo, then in the terminal:
```bash
python3 -m venv .pyenv  # Create virtual environment
source .pyenv/bin/activate  # Activate the virtual environment
# source .pyenv/Scripts/activate  # Activation if on Windows
pip install --upgrade pip  # Upgrade if old version of pip
pip install -e .  # Install requirements of the module
pip install ."[dev, test]"  # Syntax for installing optional requirements of module
```

## Prepare files for environment variables for openai
- In OpenAIs graphical interface, make sure to set right default organization
- Paste your api key in a pure file "api_key" (no file ending, dont wrap key in string quotation) inside approximating_finetuning, it should be gitignored then and read by helpers.
  - Don't add any file ending
  - Don't wrap the key in string quotation, just paste it raw into the file
  - Double check it is gitignored in your git state
- For input datasets (one should nog git commit datasets), they are available here: - https://drive.google.com/drive/folders/1TkWVsu7PVFBGEAYV-11x-yxzEx6Gt7ka?usp=sharing
  - fairytales.txt 
    - is the original childrens books dataset we use. It comes from kaggle (https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus?resource=download)
    - used for the first step which is to finetune a small model
  - (raw_api_results_{num_iterations}.pkl: Raw results from querying the openai api for the 3 models without aligning tokens. Just for reference.)
  - lp_list_{num_iterations}.pkl: This is list of logprobs after aligning them to the big model and normalizing- lp_list_{num_iterations}.pkl: This is list of logprobs after aligning them to the big model and normalizing
  - other_list_{num_iterations}.pkl: This is other metadata per text context, like the true token and what the input text was. To keep pipeline clean, its kept separate, but needed for example for computing perplexity.
  - Put it in approximating_finetuning/data/

## Use notebooks that are .py files
Assuming you've installed the dependency for Jupytext from requirements, you can then:
- Launch jupyterlab from the terminal
- Right click a .py file that is notebook --> Open with --> Jupytext Notebook. You should then see it as a regular .ipynb style notebook but the git diffs will be clean.

## Overview of the flow and notebooks
0) If starting completely from scratch, the first step is to fine tune a small model. In that case, use the execute-fine-tuning.sh bash script that prepartes the txt file into the format the tuning api wants, and starts the tuning. (Else if you have access to a tuned small model through your organization you can just use it by setting default org.) 
1) The first notebook (01-data-prep-notebook) is for:
  - Calling the openai api to get the logprobs for the 3 models for lots of examples in the test set part of the input data
  - Aligning the logprobs to the big model
  - Normalizing the logprobs
  - Dumping the results to pickle files (lp_list, other_list)
2) The second notebook (02-tuning-notebook.py) uses the hyperparamter tuning library Optuna to fine tune the input parameters for the blending pipeline to achieve the best perplexity
3) The third notebook (03-generate-example-text-notebook.py) is for generating example text using the fine tuned blending params