"""
Processes text by stemming/lemmatizing, removing stop words, and
removing punctuation. Also sentences the text.
"""

import logging
import os

import spacy

from project.utils.constants import TOKENIZED_DATA_PATH
from project.utils.functions import load_file_to_df


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 4700000


class Tokenizer:
    """
    Tokenizes text.

    parameters:
        filepath (str): path to the file to tokenize.
        filename (str): name of the file to save the tokenized data to.
    """

    def __init__(
        self,
        filepath=None,
        filename=None,
    ):
        self.tokenized_df = None
        if filepath:
            self.df = load_file_to_df(filepath)
            self.filename = filename
            self.savepath = os.path.join(TOKENIZED_DATA_PATH, self.filename)

    @staticmethod
    def tokenize_and_normalize(text: str, extra_stop: list = None) -> dict:
        """
        Tokenizes and normalizes text into sentences and words.

        parameters:
            text (str): text to tokenize and normalize.
            extra_stop (list): additional stop words to remove.

        returns:
            processed (dict): tokenized and normalized text.
        """
        if extra_stop is None:
            extra_stop = []
        processed = {"sents": [], "words": [], "words_norm": []}

        # Use spaCy for sentence segmentation and initial tokenization
        doc = nlp(text.lower())
        sentences = list(doc.sents)
        processed["sents"] = [sent.text.strip() for sent in sentences]

        for token in doc:
            # Check if token meets criteria
            if (
                not token.is_stop
                and not token.is_punct
                and not token.like_num
                and token.text.strip()
            ):
                processed["words"].append(token.text)
                processed["words_norm"].append(token.lemma_)

        return processed

    def process(self, cols_to_tokenize=None):
        """
        Processes the text.

        parameters:
            cols_to_tokenize (list): columns to tokenize.
        """
        if not cols_to_tokenize:
            cols_to_tokenize = [("cleaned_text", "tokenized_text")]

        self.tokenized_df = self.df.copy()

        for col, new_col in cols_to_tokenize:
            logging.debug(f"\tTokenizing and normalizing {col}...")
            self.tokenized_df[new_col] = (
                self.tokenized_df[col]
                .dropna()
                .apply(lambda x: self.tokenize_and_normalize(x))
            )

            # Unpack processed text into separate columns
            self.tokenized_df[f"{new_col}_sents"] = self.tokenized_df[
                new_col
            ].apply(lambda x: x["sents"])
            self.tokenized_df[f"{new_col}_words"] = self.tokenized_df[
                new_col
            ].apply(lambda x: x["words"])
            self.tokenized_df[f"{new_col}_words_norm"] = self.tokenized_df[
                new_col
            ].apply(lambda x: x["words_norm"])

            # Drop the intermediate column
            self.tokenized_df.drop(columns=[new_col], inplace=True)
