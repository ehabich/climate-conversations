"""
Models the topics discussed in Reddit submissions and comments. 

Author: Jennifer Yeaton
"""

# import nltk; nltk.download('stopwords')
import argparse
import re
import os
import sys
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pickle
import pyLDAvis

# import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR
)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def pre_process_lda(tokenized_comments):
    """
    Pre-process tokenized data ahead of running lda.

    Inputs:
     - tokenized_comments:

    Output:
     - corpus:
     - id2word:
     - data_lemmatized:
    """
    # Take column of tokenized words from their dataframe and cast to a list
    tokenized_comments_list = (
        tokenized_comments.tokenized_body_words_norm.values.tolist()
    )

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(
        tokenized_comments_list, min_count=5, threshold=100
    )  # JY: HAVE NOT YET THOUGHT ABOUT THESE NUMBERS
    trigram = gensim.models.Phrases(
        bigram[tokenized_comments_list], threshold=100
    )  # JY: HAVE NOT YET THOUGHT ABOUT THESE NUMBERS

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Form Bigrams
    data_words_bigrams = make_bigrams(bigram_mod, tokenized_comments_list)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(
        data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
    )

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus, id2word, data_lemmatized


def make_bigrams(bigram_mod, texts):
    """
    Make bigrams

    Inputs:
     - bigram_mod
     - texts

    Returns:

    """
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(trigram_mod, bigram_mod, texts):  # FLAG: YOU NEVER CALL THIS
    """
    Make bigrams

    Inputs:
     - bigram_mod
     - texts

    Returns:

    """
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """
    Lemmatize text using spacy: https://spacy.io/api/annotation

    Inputs:


    Returns:

    """
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


def run_lda(corpus, id2word):  #
    """
    Run lda on data that was pre-processed using pre_process_lda function.

    Inputs:
     - corpus
     - id2word

    Output:
     - lda_model
    """

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=20,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha="auto",
        per_word_topics=True,
    )

    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    return lda_model, doc_lda


def evaluate_lda(lda_model, data_lemmatized, id2word):
    """
    Calculate the perplexity and coherence scores of the LDA model.

    Inputs:
     - lda_model
     - data_lemmatized
     - id2word

    Returns:
     - None. Prints perplexity and coherenece scores.
    """

    # Compute Perplexity
    print(
        "\nLDA Perplexity: ", lda_model.log_perplexity(corpus)
    )  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=lda_model,
        texts=data_lemmatized,
        dictionary=id2word,
        coherence="c_v",
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print("\nLDA Coherence Score: ", coherence_lda)


def pre_process_bert():
    """
    Pre
    """


def run_bertopic():
    """
    https://maartengr.github.io/BERTopic/index.html

    https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#hierarchical-labels

    https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8
    """


def visualize_bertopic():
    """
    https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-topics-per-class


    """


def evaluation_bertopic():
    """
    Calculate the perplexity and coherence scores of the BERTopic model.

    Inputs: TBD

    Returns:
     - None. Prints perplexity and coherenece scores.

    """


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Tokenize and process text from sh file."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Path to the tokenized data to analyze.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use: either 'lda' or 'bertopic'.",
    )

    # parser.add_argument('--filename', type=str, required=True, help='Name of the file to save the tokenized data to.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Initialize logging
    # logging.basicConfig(level=logging.INFO)

    # Define parent_directory ### WAIT WILL THIS WORK???

    print("Child directory before join: ", os.getcwd())
    parent_directory = os.path.abspath(
        os.path.join(os.getcwd(), "..", "data", "tokenized")
    )
    print("Parent directory before file path creation: ", parent_directory)

    # Construct the full path to tokenized pickle file
    file_path = os.path.join(
        parent_directory,
        "tokenized_climateskeptics_sub_comments.pickle",  # CHANGE HARD CODE TO AN INPUT
    )

    print("File path prior to opening tokenized_comments is: ", file_path)

    # Try to open the file using the full path
    with open(file_path, "rb") as f:
        tokenized_comments = pickle.load(f)

    if args.model == "lda":
        corpus, id2word, data_lemmatized = pre_process_lda(tokenized_comments)
        lda_model, doc_lda = run_lda(corpus, id2word)
        evaluate_lda(lda_model, data_lemmatized, id2word)
    else:
        ____ = run_bertopic()

    # tokenizer = Tokenizer(filepath=args.filepath, filename=args.filename)
    # tokenizer.process(cols_to_tokenize=[("title", "tokenized_title")])
    # save(tokenizer.tokenized_df, tokenizer.savepath)
