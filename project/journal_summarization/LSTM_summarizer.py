"""
Implements the LSTM model for summarization

Authors: Chanteria Milner
"""

import argparse
import glob

# Packages
import json
import os
import re
import time

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
from nltk.tokenize import sent_tokenize
from rouge import Rouge
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Masking
from tensorflow.keras.callbacks import LambdaCallback

# Project imports
from project.utils.constants import (
    CLEANED_PROQUEST_FILE,
    MODEL_PATH,
    SECTION_HEADERS_MAPPING,
)
from project.utils.functions import load_file_to_df, save_df_to_file

SUMMARY_LENGTH = 10
RANDOM_STATE = 42
SAMPLE_SIZE = 100
TRAIN_SIZE = 0.7
VALID_SIZE = 0.5
EMBEDDING_DIM = 384  # 384 is the dimension of the sentence transformer model

# create encoding model
model = SentenceTransformer("all-MiniLM-L6-v2")


class LSTMSummarizer:
    """
    Class for creating summaries of climate research articles. Creates summaries
    by using the following steps:
        1. Identify article sections (Introduction, Methods, etc.) and removing
           unnecessary sections (Acknowledgements, References, etc.).
        2. Tokenize the article into sentences.
        3. Assign a weight to each sentence based on its importance. Importance
           is calculated by finding the normalized cosine similarity between
           each sentence and the article's abstract.
        4. Train the LSTM model using the tokenized sentences and their cosine
              similarities.
        5. For each article, select the top N sentences with the highest weights
           to create the summary. Order the sentences by their appearance in the
            article.

    arguments:
        file_path (str): the path to the file containing the articles.
        dev_environment (bool): whether the model is being developed.
        training (bool): whether the model is being trained.
    """

    def __init__(
        self,
        file_path=CLEANED_PROQUEST_FILE,
        dev_environment: bool = False,
        training=False,
    ):
        self.lstm_model_dict = {
            "Introduction": None,
            "Method": None,
            "Result": None,
            "Conclusion": None,
        }

        if not training:
            # the models should exist; load them
            for section in self.lstm_model_dict:
                model_path = os.path.join(MODEL_PATH, f"{section}_lstm.h5")
                self.lstm_model_dict[section] = load_model(model_path)

        self.tokenizer = None
        self.dev_environment = dev_environment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_epochs = 50 if self.dev_environment else 10
        self.batch_size = 32 if self.dev_environment else 8

        # datasets
        self.df = load_file_to_df(file_path)
        self.df = self.df.loc[
            ~(self.df.loc[:, "text"].isna())
            & (self.df.loc[:, "text"].str.len() > 0)
            & ~(self.df.loc[:, "abstract"].isna())
            & (self.df.loc[:, "abstract"].str.len() > 0),
            :,
        ]
        if self.dev_environment:
            # shuffle the data
            self.df = self.df.sample(frac=1, random_state=RANDOM_STATE).reset_index(
                drop=True
            )

            # get a random sample of the data
            self.df = self.df.sample(SAMPLE_SIZE)
            self.df.reset_index(drop=True, inplace=True)
        self.subset_df = self.df.loc[:, ["text", "abstract"]]
        self.dset = {
            "Introduction": {"Raw": [], "Abstracts": [], "Embedded": [], "Targets": []},
            "Method": {"Raw": [], "Abstracts": [], "Embedded": [], "Targets": []},
            "Result": {"Raw": [], "Abstracts": [], "Embedded": [], "Targets": []},
            "Conclusion": {"Raw": [], "Abstracts": [], "Embedded": [], "Targets": []},
        }
        self.train_test_split = {
            "Introduction": {
                "max_sentence_length": 0,
                "max_article_length": 0,
                "X_train": None,
                "y_train": None,
                "X_test": None,
                "y_test": None,
                "X_valid": None,
                "y_valid": None,
            },
            "Method": {
                "max_sentence_length": 0,
                "max_article_length": 0,
                "X_train": None,
                "y_train": None,
                "X_test": None,
                "y_test": None,
                "X_valid": None,
                "y_valid": None,
            },
            "Result": {
                "max_sentence_length": 0,
                "max_article_length": 0,
                "X_train": None,
                "y_train": None,
                "X_test": None,
                "y_test": None,
                "X_valid": None,
                "y_valid": None,
            },
            "Conclusion": {
                "max_sentence_length": 0,
                "max_article_length": 0,
                "X_train": None,
                "y_train": None,
                "X_test": None,
                "y_test": None,
                "X_valid": None,
                "y_valid": None,
            },
        }

    def extract_sections(self, article_text: str) -> dict:
        """
        Splits the text into sections.

        Args:
            text (str): text to split into sections.

        Returns:
            dict: a mapping of article text to normalized section headers, where
                  the section headers expected by the SAPGraph model are:
                  Introduction
                  Method
                  Result
                  Conclusion
        """
        # create a regex pattern for the section headers
        header_patterns = []
        for category, headers in SECTION_HEADERS_MAPPING.items():
            for header in headers:
                pattern = re.escape(header)
                header_patterns.append(pattern)
        headers_regex = (
            rf"(?:\d+\s*\.?\s*)?({'|'.join(header_patterns)})(?::?\s+)(?=[\dA-Z])"
        )

        # split the text into sections based on headers_regex
        sections = re.split(headers_regex, article_text)
        if len(sections[0].strip()) == 0:
            sections = sections[1:]
        else:
            sections = ["Introduction"] + sections

        # organize sections into a dictionary based on SECTION_HEADERS_MAPPING
        categories = list(SECTION_HEADERS_MAPPING.keys())
        prev_category = None
        normalized_sections = {category: [] for category in SECTION_HEADERS_MAPPING}
        for i in range(0, len(sections), 2):
            header, text = sections[i], sections[i + 1].strip()
            for category, headers in SECTION_HEADERS_MAPPING.items():
                # if a header comes after it already should have come
                # it is most likely a mislabeling
                if header in headers:
                    if prev_category is not None and categories.index(
                        prev_category
                    ) > categories.index(category):
                        category = "Other"
                    else:
                        prev_category = category
                    normalized_sections[category].extend(sent_tokenize(text))
                    break

        normalized_sections.pop("Other", None)

        return normalized_sections

    def grab_section_data(self, row: pd.Series, dset: dict) -> list:
        """
        Grabs the section data from the article.

        params:
            row (pd.Series): a row from the dataframe including the article
            abstract and sectioned text.
            dset (dict): the dataset to add the section data to.
        """
        article_sections = row["text_sections"]
        abstract = row["abstract"]

        for section, sentences in article_sections.items():
            dset[section]["Raw"].append(sentences)
            dset[section]["Abstracts"].append(abstract)

    def preprocess(self) -> None:
        """
        Preprocesses the data for in preparation for the LSTM summarization.
        """
        print("Preprocessing data...")

        # extract sections
        print("\tGetting text sections...")
        self.subset_df["text_sections"] = self.subset_df["text"].apply(
            self.extract_sections
        )
        self.subset_df.dropna(inplace=True)

        # add to the dataset
        print("\tAdding to dataset...")
        self.subset_df.loc[:, ["text_sections", "abstract"]].apply(
            lambda x: self.grab_section_data(x, self.dset), axis=1
        )

    def get_targets(self, abstracts, sentences) -> None:
        """
        Gets the targets for the LSTM model by calculating the cosine
        similarities between the passed sentences and the abstract

        parameters:
            abstracts (list): a list of abstracts.
            sentences (list): a list of embedded sentences.
        """
        all_targets = []
        for abstract, sents in zip(abstracts, sentences):
            abstract_vector = model.encode(abstract)
            targets = []
            for sentence in sents:
                # exclude the first element, which is the normalized sentence order
                sim = cosine_similarity([sentence[1:]], [abstract_vector])[0][0]
                targets.append(sim)
            all_targets.append(targets)

        return all_targets

    def prepare_datasets(self) -> None:
        """
        Prepares the datasets for training the LSTM model.
        """
        print("Preparing datasets...")

        # embedd the sentences in each article
        for section in self.dset:

            # embedd article sentences
            embedded_articles = [
                [model.encode(sentence) for sentence in article]
                for article in self.dset[section]["Raw"]
            ]

            # add normalized sentence order to embeddings
            for article in embedded_articles:
                for i, embedding in enumerate(article):
                    norm_order = np.array((i + 1) / len(article))
                    article[i] = np.append(norm_order, embedding)

            # pad article sections to have a uniform length
            max_article_length = max(len(article) for article in embedded_articles)
            max_sentence_length = max(
                len(sentence) for article in embedded_articles for sentence in article
            )
            padded_articles = pad_sequences(
                [np.array(article) for article in embedded_articles],
                maxlen=max_article_length,
                padding="post",
                dtype="float32",
                value=0.0,
            )
            self.dset[section]["Embedded"] = padded_articles

            # save the max sentence and article length
            self.train_test_split[section]["max_sentence_length"] = max_sentence_length
            self.train_test_split[section]["max_article_length"] = max_article_length

            # get the targets
            self.dset[section]["Targets"] = self.get_targets(
                self.dset[section]["Abstracts"], self.dset[section]["Embedded"]
            )

            # train-test-valid split
            X_train, X_test, y_train, y_test = train_test_split(
                self.dset[section]["Embedded"],
                self.dset[section]["Targets"],
                train_size=TRAIN_SIZE,
                random_state=RANDOM_STATE,
            )
            X_test, X_valid, y_test, y_valid = train_test_split(
                X_test, y_test, train_size=VALID_SIZE, random_state=RANDOM_STATE
            )
            self.train_test_split[section]["X_train"] = X_train
            self.train_test_split[section]["y_train"] = y_train
            self.train_test_split[section]["X_test"] = X_test
            self.train_test_split[section]["y_test"] = y_test
            self.train_test_split[section]["X_valid"] = X_valid
            self.train_test_split[section]["y_valid"] = y_valid

    def train_lstm_model(self, section: str) -> None:
        """
        Trains the LSTM model for a given category.

        parameters:
            section (str): the section to train the model for.
        """
        # for printing progress
        print_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"\t\tEpoch {epoch + 1}: Training Loss: {logs['loss']}, Validation Loss: {logs['val_loss']}"
            )
        )

        # pull the data
        X_train = self.train_test_split[section]["X_train"]
        y_train = self.train_test_split[section]["y_train"]
        X_valid = self.train_test_split[section]["X_valid"]
        y_valid = self.train_test_split[section]["y_valid"]
        X_test = self.train_test_split[section]["X_test"]
        y_test = self.train_test_split[section]["y_test"]

        # build the model
        input_shape = (
            None,
            EMBEDDING_DIM + 1,
        )  # 'None' allows for variable sentence counts

        inputs = Input(shape=input_shape)
        masked = Masking(mask_value=0.0)(inputs)  # 0.0 is the padding value
        lstm_out = LSTM(units=50, return_sequences=True)(
            masked
        )  # return_sequences=True to keep sentence-level outputs
        sentence_importance = TimeDistributed(Dense(1, activation="linear"))(
            lstm_out
        )  # assuming binary importance score; adjust activation for your task

        model = Model(inputs=inputs, outputs=sentence_importance)
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.summary()

        # train model
        history = model.fit(
            X_train,
            y_train,
            epochs=self.training_epochs,
            validation_data=(X_valid, y_valid),
            batch_size=self.batch_size,
            callbacks=[print_callback],
        )

        # evaluate the model
        test_loss = model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}")

        # save the model
        model.save(os.path.join(MODEL_PATH, f"{section}_lstm.h5"))

    def train_lstm(self) -> None:
        """
        Trains the LSTM models. There are four models, one for each section
        of interest.
        """
        print("Training LSTM models...")

        # train the LSTM models
        for section in self.train_test_split:
            print(f"\tTraining {section} model...")
            self.train_lstm_model(section)

    def summarize(self, article_text: str, top_n_per_section: int = 2) -> str:
        """
        Summarizes an article by selecting the top N sentences from each
        relevant section.

        Parameters:
            article_text (str): The text of the article to summarize.
            top_n_per_section (int): The number of top sentences to select from
                                     each section for the summary.

        Returns:
            str: The summarized text.
        """
        # extract sections from the article
        sections = self.extract_sections(article_text)
        summary_sentences_with_order = []
        sentences_selected = 0

        # process each section
        for section, sentences in sections.items():
            if section not in self.lstm_model_dict or not sentences:
                continue  # skip sections without a model or sentences

            # embed sentences and add normalized sentence order
            embedded_sentences = [
                np.append(np.array([(i + 1) / len(sentences)]), model.encode(sentence))
                for i, sentence in enumerate(sentences)
            ]

            # pad the sequences for the model
            padded_sentences = pad_sequences(
                [np.array(embedded_sentences)],
                maxlen=self.train_test_split[section]["max_article_length"],
                padding="post",
                dtype="float32",
                value=0.0,
            )

            # predict importance scores
            importance_scores = self.lstm_model_dict[section].predict(padded_sentences)[
                0
            ]

            # calculate remaining sentences to select based on total limit
            remaining_sentences_to_select = SUMMARY_LENGTH - sentences_selected
            top_n = min(top_n_per_section, remaining_sentences_to_select)

            # select top N sentences based on the scores,
            # not exceeding total summary length
            top_indices = np.argsort(-importance_scores[:, 0])[:top_n]
            for i in top_indices:
                if i < len(sentences) and sentences_selected < SUMMARY_LENGTH:
                    # store the sentence and its order
                    summary_sentences_with_order.append(
                        (sentences[i], embedded_sentences[i][0])
                    )
                sentences_selected += 1
                if sentences_selected >= SUMMARY_LENGTH:
                    break  # stop adding sentences once limit reached

        # sort the sentences by their normalized orders before compiling the
        # summary
        summary_sentences_with_order.sort(key=lambda x: x[1])
        summary_sentences = [sentence for sentence, _ in summary_sentences_with_order]

        # combine selected sentences to form the summary
        summary = " ".join(summary_sentences)
        return summary
