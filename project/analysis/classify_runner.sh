#!/bin/bash
#SBATCH --mail-user=jkyeaton@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=dsi_climate
#SBATCH --output=reddit_token_output/%j.%N.stdout
#SBATCH --error=reddit_token_output/%j.%N.stderr
#SBATCH --chdir=/home/jkyeaton/dsi_files_addtl/climate-conversations/project
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate climate_env

export INPUT_DATA_FILE_PATH="project/data_collection/project_data/tokenized_climate_comments.pickle"
export COL_TO_TOKENIZE='Body'
export SUBREDDIT='None'
export TOKENIZE='False'
export TYPE='comment'

# Pass vars to the python script
python3 moral_strength_classifier.py --filepath ${INPUT_DATA_FILE_PATH} --col_to_tokenize ${COL_TO_TOKENIZE} --subreddit ${SUBREDDIT} --tokenize ${TOKENIZE} --type ${TYPE}
