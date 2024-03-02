import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


# EXAMPLE CLI RUN
# poetry run python moral_strength_classifier.py --filepath project/data_collection/project_data/tokenized_climate_comments.pickle --col_to_tokenize body --token_name tokenized_body --subreddit climate --utc_start 1662654781 --utc_end 1662703120 --rows 0,2 --tokenize False

print("Finished Package Imports!")

parent_directory = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(parent_directory)

from project.utils.classes.tokenizer import Tokenizer


if os.path.exists("wordvectors.kv"):
    word_vectors = KeyedVectors.load("wordvectors.kv")
else:
    import gensim.downloader as api

    word_vectors = api.load("glove-twitter-200")
    word_vectors.save("wordvectors.kv")

print("Loaded Word Vectors!")

def load_data(filepath, subreddit=None, utc_start=None, utc_end=None, rows=None, col_to_tokenize=None):
    pickle_file_path_comment = os.path.join(
        parent_directory,
        filepath,
    )

    with open(pickle_file_path_comment, "rb") as file:
        comments_df = pickle.load(file)

    if subreddit:
        comments_df = comments_df[comments_df["subreddit"] == subreddit]

    if utc_start and utc_end:
        comments_df = comments_df[
            (comments_df["created_utc"] >= utc_start)
            & (comments_df["created_utc"] <= utc_end)
        ]

    try:
        comments_df = comments_df[
            ~comments_df['tokenized_body_words_norm'].isin(["[removed]", "[deleted]"])
        ]
    except:
        print('Could not find removed or deleted entries')

    if rows:
        start_row, end_row = rows
        comments_df = comments_df.iloc[start_row:end_row]

    if subreddit:
        print(f"Filtered data for {subreddit}!")

    else:
        print('Filtered data!')
    return (comments_df)


def tokenize_comments(df, subreddit, col_to_tokenize, token_name):
    pickle_path = f"comments_{subreddit}.pkl"
    token_pickle_path = f"tokenized_comments_{subreddit}.pkl"
    df.to_pickle(pickle_path)
    tokenizer = Tokenizer(filepath=pickle_path, filename=token_pickle_path)
    tokenizer.df = df
    tokenizer.process(cols_to_tokenize=[(col_to_tokenize, token_name)])
    print("Tokenizer Complete!")
    return tokenizer.tokenized_df


def compute_similarity(
    comment, foundation_words_vec, similarity_threshold=0.25
):
    similarities = []
    for word in comment:  # word in reddit comment
        try:
            word_vec = word_vectors[word]  # get the embedding
            for (
                foundation_word_vec
            ) in foundation_words_vec:  # loop through moral foundation words
                sim = np.dot(word_vec, foundation_word_vec) / (
                    np.linalg.norm(word_vec)
                    * np.linalg.norm(foundation_word_vec)
                )
                # Apply threshold
                if sim >= similarity_threshold:
                    similarities.append(sim)
        except KeyError:  # If the word is not in the embedding vocabulary
            pass

    if similarities:
        return np.mean(similarities)
    else:
        return 0


def classify_sentence_with_profile(sentence, moral_foundations_dict):
    foundation_scores = {}

    for foundation, words in moral_foundations_dict.items():
        words_vec = []
        for word in words:
            try:  # loop through moral foundation words
                word_vec = word_vectors[word]
                words_vec.append(word_vec)
            except:
                pass
        foundation_scores[foundation] = compute_similarity(sentence, words_vec)

    return foundation_scores


def main(filepath, subreddit=None, utc_start=None, utc_end=None, rows=None, tokenize=False, col_to_tokenize=False, token_name='tokenized_body'):

    comments_df = load_data(
        filepath=filepath, subreddit=subreddit, utc_start=utc_start, utc_end=utc_end, rows=rows
    )

    if tokenize is True:
        tokenized_comments = tokenize_comments(comments_df, subreddit, col_to_tokenize, token_name)
    else:
        print('Columns Already Tokenized!')
        tokenized_comments = comments_df

    with open("expanded_moral_foundations_dictionary.json", "r") as f:
        word_to_moral_foundation_expanded = json.load(f)

    # classify sentences
    classification_profiles = []
    for comment in tokenized_comments['tokenized_body_words_norm']:
        classification_profile = classify_sentence_with_profile(
            comment, word_to_moral_foundation_expanded
        )
        classification_profiles.append(classification_profile)
    print("Finished Classification!")

    # add classifications to original df
    df_classification_profiles = pd.DataFrame(classification_profiles)
    result_df = pd.concat(
        [
            tokenized_comments.reset_index(drop=True),
            df_classification_profiles.reset_index(drop=True),
        ],
        axis=1,
    )

    # write to pickle file
    start_row, end_row = rows if rows else (None, None)
    filename = (
        f"result_{subreddit}_{utc_start}_{utc_end}_{start_row}_{end_row}.pkl"
    )
    result_df.to_pickle(filename)
    print(f"Saved DataFrame to {filename}!")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process comments or submissions.")
    
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Filepath to data",
    )

    parser.add_argument(
        "--col_to_tokenize",
        type=str,
        help="Column to tokenize.",
    )

    parser.add_argument(
        "--token_name",
        type=str,
        help="Output name for tokenized column. Only needed if tokenize = True",
    )

    parser.add_argument(
        "--subreddit", type=str, help="Subreddit name"
    )
    parser.add_argument(
        "--utc_start", type=int, help="Start time in UTC"
    )
    parser.add_argument(
        "--utc_end", type=int, help="End time in UTC"
    )
    parser.add_argument(
        "--rows",
        type=str,
        help="Starting and ending row indices separated by comma",
    )
    parser.add_argument(
        "--tokenize",
        type=str,
        required=True,
        help="True to tokenize data before classifying, false if data is already tokenized",
    )

    args = parser.parse_args()

    # Handling optional string arguments that could be 'None'
    args.subreddit = None if args.subreddit == 'None' else args.subreddit
    args.col_to_tokenize = None if args.col_to_tokenize == 'None' else args.col_to_tokenize
    args.token_name = None if args.token_name == 'None' else args.token_name

    # Handling the 'rows' argument
    if args.rows and args.rows.lower() != 'none':
        rows = tuple(map(int, args.rows.split(",")))
    else:
        rows = None

    # Convert tokenize argument from string to boolean
    tokenize = args.tokenize.lower() in ['true', '1', 't', 'y', 'yes']

    result_df = main(
        filepath=args.filepath,
        col_to_tokenize=args.col_to_tokenize,
        token_name=args.token_name,
        subreddit=args.subreddit,
        utc_start=args.utc_start,
        utc_end=args.utc_end,
        rows=rows,
        tokenize=tokenize
    )
    print(result_df.head())
