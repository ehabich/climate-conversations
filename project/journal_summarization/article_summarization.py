"""
Implement SAPGraph model for article summarization.
Fine-Tune on journal articles --> abstracts.

Authors: Chanteria Milner, Kate Habich
Credit: https://github.com/cece00/SAPGraph/tree/main?tab=readme-ov-file
"""

# Packages
import re
import os
import pandas as pd
import json
from nltk.tokenize import sent_tokenize

# External imports

# Project imports
from project.utils.classes.tokenizer import Tokenizer
from project.utils.functions import load_file_to_df, save_df_to_file
from project.utils.constants import (
    TOKENIZED_DATA_PATH,
    CLEANED_DATA_PATH,
    CLEANED_PROQUEST_FILE,
    SECTION_HEADERS_MAPPING,
)


class ArticleSummarizer:
    """
    Summarizes journal articles. Adapts the SAPGraph model for article
    summarization.

    arguments:
        file_path (str): path to the file to summarize.
        dev_environment (bool): whether or not the environment is a development
                                environment.
    """

    def __init__(self, file_path: str, dev_environment: bool = False):
        self.df = load_file_to_df(file_path)

        # used for model building
        if dev_environment:
            # shuffle the data
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

            # get a random sample of the data
            self.df = self.df.sample(100)

        self.subset_df = self.df.loc[:, ["text", "abstract"]]
        self.tokenizer = Tokenizer()

    def section_text(self, article_text: str) -> dict:
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
        # create a regex pattern to split the text into sections
        pattern_headers = "|".join(
            re.escape(header) for header in SECTION_HEADERS_MAPPING.keys()
        )
        headers_regex = rf"(?:\d+\s*\.?\s*)?({pattern_headers})(?::?\s+)(?=[\dA-Z])"

        # split the text into sections
        sections = re.split(headers_regex, article_text)
        if len(sections[0].strip()) > 0:
            sections = ["Introduction"] + sections
        else:
            sections = sections[1:]
        sections = [
            (sections[i], sections[i + 1].strip()) for i in range(0, len(sections), 2)
        ]

        # no sections found in article text
        if len(sections) == 0:
            return None

        # normalize section headers
        normalized_sections = {
            "Introduction": [],
            "Method": [],
            "Result": [],
            "Conclusion": [],
            "Other": [],
        }
        for header, text in sections:
            normalized_header = SECTION_HEADERS_MAPPING.get(header, "Other")
            normalized_sections[normalized_header].extend(sent_tokenize(text))

        return normalized_sections

    @staticmethod
    def row_to_json(row: pd.Series) -> dict:
        """
        Converts a row of the dataframe to a dictionary.

        Args:
            row (pd.Series): row of the dataframe to convert to a dictionary.

        Returns:
            dict: dictionary representation of the row.
        """
        textlist = [
            [section] + sentences
            for section, sentences in row["normalized_text"].items()
        ]
        return json.dumps(
            {
                "text": textlist,
                "summary": sent_tokenize(row["abstract"]),
                "label": [],
                "entity": [],
            }
        )

    def save_json(self) -> None:
        """
        Saves the data to a json file.

        Args:
            file_path (str): path to save the data to.
        """
        savepath = os.path.join(CLEANED_DATA_PATH, f"proquest_data_sapgraph.jsonl")
        with open(savepath, "w", encoding="utf8") as f:
            for _, row in self.subset_df.loc[
                :, ["normalized_text", "abstract"]
            ].iterrows():
                json_line = self.row_to_json(row)
                f.write(json_line + "\n")

    def preprocess_data(self) -> None:
        """
        Preprocesses the data for summarization.
        """
        # grab only the rows where the text and abstract are not null
        self.subset_df.dropna(inplace=True)
        self.subset_df = self.subset_df.loc[
            (self.subset_df.loc[:, "text"].apply(len) > 0)
            & self.subset_df.loc[:, "abstract"].apply(len)
            > 0,
            :,
        ]

        self.subset_df.reset_index(drop=True, inplace=True)

        # split text into sections
        self.subset_df["normalized_text"] = self.subset_df["text"].apply(
            self.section_text
        )
        self.subset_df.dropna(inplace=True)

        # save the data to a json file
        self.save_json()

    def train_model(self) -> None:
        """
        Trains the model on the data.
        """
        pass

    def summarize_articles(self) -> None:
        """
        Summarizes the articles in the dataframe.
        """
        pass

    def save_summaries(self, file_path: str) -> None:
        """
        Saves the summaries to the specified file path.
        """
        save_df_to_file(self.df, file_path)
