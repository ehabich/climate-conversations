
"""
The module is designed to take reddit comments, optionally tokenize them,
and then evaluate the similarity of the comment with each of the 5 moral
foundations. It returns an updated dataframe with moral foundation similarity
scores.

Author: Kathryn Link-Oberstar
"""
import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

parent_directory = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(parent_directory)

from project.utils.classes.tokenizer import Tokenizer

# Download word embeddings, or retrieve saved embeddings if they exist
if os.path.exists("wordvectors.kv"):
    word_vectors = KeyedVectors.load("wordvectors.kv")
else:
    import gensim.downloader as api
    word_vectors = api.load("glove-twitter-200")
    word_vectors.save("wordvectors.kv")
print("Loaded Word Vectors!")

# Load the data, unpickle and filter if necessary
def load_data(
    filepath,
    subreddit=None,
    col_to_tokenize=None,
):
    """
    Loads and filters reddit comments data from a pickle file.

    Args:
        filepath (str): Path to the pickle file containing comments data.
        subreddit (str, optional): Subreddit name to filter comments. Defaults to None.
        col_to_tokenize (str, optional): Column name for tokenization. Defaults to None.

    Returns:
        DataFrame: Filtered dataframe with comments data.
    """
    pickle_file_path_comment = os.path.join(
        parent_directory,
        filepath,
    )

    with open(pickle_file_path_comment, "rb") as file:
        comments_df = pickle.load(file)

    if subreddit:
        comments_df = comments_df[comments_df["subreddit"] == subreddit]

    try:
        token_key = f"tokenized_{col_to_tokenize.lower()}_words_norm"
        comments_df = comments_df[
            ~comments_df[token_key].isin(["[removed]", "[deleted]"])
        ]
    except:
        print("Could not find removed or deleted entries")

    if subreddit:
        print(f"Filtered data for {subreddit}!")
    else:
        print("Filtered data!")
    
    return comments_df


# Tokenize the comments if the argument is specified
def tokenize_comments(df, subreddit, col_to_tokenize):
    """
    Tokenizes comments within the dataframe using specified column.

    Args:
        df (DataFrame): Dataframe containing comments to be tokenized.
        subreddit (str): Name of the subreddit for tokenization scope.
        col_to_tokenize (str): Column name in the dataframe to tokenize.

    Returns:
        DataFrame: Dataframe with tokenized comments.
    """
    pickle_path = f"comments_{subreddit}.pkl"
    token_pickle_path = f"tokenized_comments_{subreddit}.pkl"
    df.to_pickle(pickle_path)
    tokenizer = Tokenizer(filepath=pickle_path, filename=token_pickle_path)
    tokenizer.df = df
    tokenizer.process(
        cols_to_tokenize=[
            (col_to_tokenize, "tokenized_" + col_to_tokenize.lower())
        ]
    )
    print("Tokenizer Complete!")
    return tokenizer.tokenized_df

# Compute the similarity of a comment with the 5 moral foundations
def compute_similarity(
    comment, foundation_words_vec, similarity_threshold=0.25
):
    """
    Computes the similarity of each word in a comment with words in moral foundations.

    Args:
        comment (list): List of words in a comment.
        foundation_words_vec (list): List of word vectors representing moral foundations.
        similarity_threshold (float, optional): Threshold for counting similarity. Defaults to 0.25.

    Returns:
        float: Mean similarity score for the comment with moral foundations.
    """
    similarities = []
    for word in comment:  # word in reddit comment
        try:
            word_vec = word_vectors[word]  # get embedding
            for (
                foundation_word_vec
            ) in foundation_words_vec:  # loop through moral foundation words
                sim = np.dot(word_vec, foundation_word_vec) / (
                    np.linalg.norm(word_vec)
                    * np.linalg.norm(foundation_word_vec)
                )
                # Check if similarity above threshold to reduce noise
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
    """
    Computes similarity scores between a sentence and various moral foundations.

    Args:
        sentence (str): Sentence to classify
        moral_foundations_dict (dict): {moral foundation: [keywords]} mapping

    Returns:
        dict: {moral foundation: similarity score} for each foundation.
    """
    for foundation, words in moral_foundations_dict.items():
        words_vec = []
        for word in words:
            try:
                word_vec = word_vectors[word]
                words_vec.append(word_vec)
            except:
                pass
        foundation_scores[foundation] = compute_similarity(sentence, words_vec)

    return foundation_scores


def main(
    filepath,
    subreddit=None,
    tokenize=False,
    col_to_tokenize=False,
    type="Undefined",
):
    """
    Main function to load, tokenize, and classify comments or submissions.

    Args:
        filepath (str): Path to data file.
        subreddit (str, optional): Subreddit name. Defaults to None.
        tokenize (bool, optional): Whether to tokenize the data. Defaults to False, assuming data was already tokenized.
        col_to_tokenize (str, optional): Column to tokenize. Defaults to False.
        type (str, optional): Type of the data (comment or submission). Defaults to "Undefined".

    Returns:
        DataFrame: Dataframe with classified comments or submissions.
    """
    comments_df = load_data(
        filepath=filepath,
        subreddit=subreddit,
    )

    if tokenize is True:
        tokenized_comments = tokenize_comments(
            comments_df, subreddit, col_to_tokenize
        )
    else:
        print("Columns Already Tokenized!")
        tokenized_comments = comments_df

    with open("expanded_moral_foundations_dictionary.json", "r") as f:
        word_to_moral_foundation_expanded = json.load(f)

    # classify sentences
    classification_profiles = []
    token_key = f"tokenized_{col_to_tokenize.lower()}_words_norm"
    for comment in tokenized_comments[token_key]:
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
    filename = f"result_{subreddit}_{type}.pkl"
    result_df.to_pickle(filename)
    print(f"Saved DataFrame to {filename}!")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process comments or submissions."
    )

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

    parser.add_argument("--subreddit", type=str, help="Subreddit name")

    parser.add_argument(
        "--type", type=str, help="Token type (i.e. submission, comment)"
    )

    parser.add_argument(
        "--tokenize",
        type=str,
        required=True,
        help="True to tokenize data before classifying, false if data is already tokenized",
    )

    args = parser.parse_args()

    # Handling optional string arguments that could be 'None'
    args.subreddit = None if args.subreddit == "None" else args.subreddit
    args.col_to_tokenize = (
        None if args.col_to_tokenize == "None" else args.col_to_tokenize
    )
    # check for truthy values
    if args.tokenize.lower() in ["true", "1", "t", "y", "yes"]:
        tokenize = True
    else:
        tokenize = False
        
    if args.tokenize.lower() in ["true", "1", "t", "y", "yes"]:
        tokenize = True
    else:
        tokenize = False

    result_df = main(
        filepath=args.filepath,
        col_to_tokenize=args.col_to_tokenize,
        subreddit=args.subreddit,
        tokenize=tokenize,
        type=args.type,
    )
    print(result_df.head())
