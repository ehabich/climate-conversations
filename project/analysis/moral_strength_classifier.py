import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import gensim.downloader as api
import argparse
from datetime import datetime

print('Finished Imports')

parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(parent_directory)

from project.utils.classes.tokenizer import Tokenizer
from moralstrength.moralstrength import string_moral_values

word_vectors = api.load("glove-twitter-200")

print('Loaded Word Vectors')

def load_data(subreddit=None, utc_start=None, utc_end=None, rows=None):
    
    pickle_file_path_comment = os.path.join(parent_directory, 'project/data_collection/project_data/climateCommentsDf.pickle')
    
    with open(pickle_file_path_comment, 'rb') as file:
        comments_df = pickle.load(file)
    
    # Filter by subreddit if specified
    if subreddit:
        comments_df = comments_df[comments_df['subreddit'] == subreddit]
    
    # Filter by UTC timestamps if specified
    if utc_start and utc_end:
        comments_df = comments_df[(comments_df['created_utc'] >= utc_start) & (comments_df['created_utc'] <= utc_end)]
    
    # Filter out '[removed]' and '[deleted]' comments
    comments_df = comments_df[~comments_df['body'].isin(['[removed]', '[deleted]'])]
    
    if rows:
        start_row, end_row = rows
        comments_df = comments_df.iloc[start_row:end_row]
    print(f'Filtered data for {subreddit}')
    return comments_df

def tokenize_comments(df):
    df.to_csv('comments.csv', index=False)
    tokenizer = Tokenizer(filepath='comments.csv', filename='tokenized_comments.csv')
    tokenizer.df = df
    tokenizer.process(cols_to_tokenize=[('body', 'tokenized_body')])
    print('Tokenizer Complete')
    return tokenizer.tokenized_df

def compute_similarity(comment, foundation_words):
    similarities = []
    for word in comment:  # word in reddit comment
        if word in word_vectors:  # check for presence in embeddings
            word_vec = word_vectors[word]  # get the embedding
            for foundation_word in foundation_words:  # loop through moral foundation words
                if foundation_word in word_vectors:  # check for presence in embeddings
                    foundation_word_vec = word_vectors[foundation_word]  # get the embedding
                    sim = np.dot(word_vec, foundation_word_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(foundation_word_vec))
                    similarities.append(sim)
    if similarities:
        return np.mean(similarities) 
    else:
        return 0

def classify_sentence_with_profile(sentence, moral_foundations_dict):
    foundation_scores = {}

    for foundation, words in moral_foundations_dict.items():
        foundation_scores[foundation] = compute_similarity(sentence, words)

    return foundation_scores


def main(subreddit=None, utc_start=None, utc_end=None, rows=None):
    comments_df = load_data(subreddit=subreddit, utc_start=utc_start, utc_end=utc_end, rows=rows)
    tokenized_comments = tokenize_comments(comments_df)
    
    with open('expanded_moral_foundations_dictionary.json', 'r') as f:
        word_to_moral_foundation_expanded = json.load(f)
    
    # Classify each comment
    classification_profiles = []
    for comment in tokenized_comments['tokenized_body_words_norm']:
        classification_profile = classify_sentence_with_profile(comment, word_to_moral_foundation_expanded)
        classification_profiles.append(classification_profile)
    print('Finished Classification')
    # Convert classification profiles to DataFrame and merge with tokenized comments
    df_classification_profiles = pd.DataFrame(classification_profiles)
    result_df = pd.concat([tokenized_comments.reset_index(drop=True), df_classification_profiles.reset_index(drop=True)], axis=1)
    
    start_row, end_row = rows if rows else (None, None)
    filename = f"result_{subreddit}_{utc_start}_{utc_end}_{start_row}_{end_row}.csv"
    result_df.to_csv(filename, index=False)
    print(f"Saved DataFrame to {filename}")
    
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process subreddit comments.')
    parser.add_argument('--subreddit', type=str, required=True, help='Subreddit name')
    parser.add_argument('--utc_start', type=int, required=True, help='Start time in UTC')
    parser.add_argument('--utc_end', type=int, required=True, help='End time in UTC')
    parser.add_argument('--rows', type=str, help='Starting and ending row indices separated by comma')

    args = parser.parse_args()

    # Convert rows argument from string to tuple of integers
    if args.rows:
        rows = tuple(map(int, args.rows.split(',')))
    else:
        rows = None

    result_df = main(subreddit=args.subreddit, utc_start=args.utc_start, utc_end=args.utc_end, rows=rows)
    print(result_df.head())

# poetry run python moral_strength_classifier.py --subreddit worldnews --utc_start 1662654781 --utc_end 1662703120 --rows 0,200
