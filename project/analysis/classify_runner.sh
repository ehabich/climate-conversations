#!/bin/bash

# This script is designed to run the moral_strength_classifier.py with different subreddits.
# Ensure the Python script is executable and properly configured to handle the provided arguments.
# Usage:
# chmod +x classify_runner.sh         # Make the script executable
# nohup ./classify_runner.sh &        # Execute the script
# ps aux | grep classify_runner.sh    # Check status of execution
# tail -f nohup.out                   # Check nohup output

# Activate the Poetry virtual environment
VENV_PATH=$(poetry env info -p)
source "$VENV_PATH/bin/activate"

BASE_COMMAND="python moral_strength_classifier.py"

# Run each command in the background
$BASE_COMMAND --subreddit worldnews &   # Run in background
$BASE_COMMAND --subreddit climateskeptics &   # Run in background
$BASE_COMMAND --subreddit climate &   # Run in background
$BASE_COMMAND --subreddit environment &   # Run in background
$BASE_COMMAND --subreddit climatechange &   # Run in background
$BASE_COMMAND --subreddit climateOffensive &   # Run in background
$BASE_COMMAND --subreddit science &   # Run in background
$BASE_COMMAND --subreddit politics &   # Run in background

# Wait for all background processes to finish
wait

echo "Processing all subreddits complete."