printf "WARNING - The tuning will use whatever default organization you have set in OpenAI GUI"

printf "\nSetting your OpenAI API key from file api_key"
export OPENAI_API_KEY=$(cat api_key)

# Assumes you have right python environment set up
printf "\nPreparing csv input file using python script"
python prepare_tuning.py

printf "\nPreparing data for fine-tuning with OpenAI tools"
openai tools fine_tunes.prepare_data -f 'data/jan_style_tuning_raw.csv'

printf "\nStarting fine tuning with assumption you have set right default organization in openai GUI"
openai api fine_tunes.create -t "data/jan_style_tuning_raw_prepared.jsonl" -m 'babbage'
