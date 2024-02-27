"""
Implement SAPGraph model for article summarization.
Fine-Tune on journal articles --> abstracts.

Authors: Chanteria Milner, Kate Habich
Credit: https://github.com/cece00/SAPGraph/tree/main?tab=readme-ov-file
"""

import argparse

# Packages
import json
import os
import re

import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize

from external_code.SAPGraph.script.calEdge import main, main_train

# external packages
from external_code.SAPGraph.script.createVoc import catDoc, getEnt
from external_code.SAPGraph.script.lowTFIDFWords import calTFidf

# Project imports
from project.utils.classes.tokenizer import Tokenizer
from project.utils.constants import (
    CLEANED_DATA_PATH,
    CLEANED_PROQUEST_FILE,
    PUNCTUATION_FILTER,
    SECTION_HEADERS_MAPPING,
)
from project.utils.functions import load_file_to_df, save_df_to_file


SAMPLE_SIZE = 100
RANDOM_STATE = 123


class ArticleSummarizer:
    """
    Summarizes journal articles. Adapts the SAPGraph model for article
    summarization.

    arguments:
        file_path (str): path to the file to summarize.
        dev_environment (bool): whether or not the environment is a development
                                environment.
    """

    def __init__(
        self,
        file_path: str = CLEANED_PROQUEST_FILE,
        dev_environment: bool = False,
    ):
        self.df = load_file_to_df(file_path)
        self.preprocessed_data_path = os.path.join(
            CLEANED_DATA_PATH, "proquest", "proquest_data_sapgraph.jsonl"
        )
        self.dataset = "proquest"
        self.dev_environment = dev_environment

        # make data directory if necessary
        if not os.path.exists(os.path.join(CLEANED_DATA_PATH, "proquest")):
            os.makedirs(os.path.join(CLEANED_DATA_PATH, "proquest"))

        # used for model building
        if self.dev_environment:
            # shuffle the data
            self.df = self.df.sample(
                frac=1, random_state=RANDOM_STATE
            ).reset_index(drop=True)

            # get a random sample of the data
            self.df = self.df.sample(SAMPLE_SIZE)
            self.df.reset_index(drop=True, inplace=True)

        self.subset_df = self.df.loc[:, ["text", "abstract"]]
        self.tokenizer = Tokenizer()
        self.NLP = spacy.load("en_core_sci_md")  # the one the article uses

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
        # create a regex pattern to split the text into sections
        pattern_headers = "|".join(
            re.escape(header) for header in SECTION_HEADERS_MAPPING.keys()
        )
        headers_regex = (
            rf"(?:\d+\s*\.?\s*)?({pattern_headers})(?::?\s+)(?=[\dA-Z])"
        )

        # split the text into sections
        sections = re.split(headers_regex, article_text)
        if len(sections[0].strip()) > 0:
            sections = ["Introduction"] + sections
        else:
            sections = sections[1:]
        sections = [
            (sections[i], sections[i + 1].strip())
            for i in range(0, len(sections), 2)
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

    def extract_entities(self, list_of_sentences: list) -> list:
        """
        Extracts the entities from the list of sentences using SciSpacy.

        Args:
            list_of_sentences (list): list of sentences to extract entities
            from.

        Returns:
            list: list of entities.
        """
        entities = []
        for sentence in list_of_sentences:
            doc = self.NLP(sentence)
            ents = []
            for ent in doc.ents:
                ents.append([ent.start, ent.end, ent.text])
            entities.append(ents)
        return entities

    def extract_text_components(self, row: pd.Series) -> list:
        """
        Extracts the text components from the text, including the text
        (broken down into sections and sentences per section), the entities,
        and the entity types.

        parameters:
            row (pd.Series): row of the dataframe to extract text components
                            from.

        Returns:
            dict: dictionary of text components.

        For each article, the data should be in the following form for SAPGraph:
            section: [section, section, ...]
            text: [[sentence, sentence, ...], [sentence, sentence, ...], ...]
                - sentences within each section
            entity: [[[entity, entity, ...], [entity, entity, ...]],
                    [[entity, entity, ...], [entity, entity, ...]],, ...]
                - entities within each sentence
            summary: [sentence, sentence, ...]
                - the article abstract or summary
        """
        text = row["text"]
        summary = row["abstract"]

        # extract text sections
        sectioned_text = self.extract_sections(text)
        text = []
        section = []
        entities = []

        # extract entities, sections, and text
        if sectioned_text is not None:
            for sec, sentences in sectioned_text.items():
                section.append(sec)
                text.append(sentences)
                ents = self.extract_entities(sentences)
                entities.append(ents)
        else:
            return None

        return {
            "section_name": section,
            "text": text,
            "entity": entities,
            "summary": sent_tokenize(summary),  # "abstract" in the dataframe
        }

    def save_json(self) -> None:
        """
        Saves the data to a json file.

        Args:
            file_path (str): path to save the data to.
        """
        with open(self.preprocessed_data_path, "w", encoding="utf8") as f:
            for _, row in self.subset_df.loc[:, ["text_entities"]].iterrows():
                entities = row["text_entities"]
                ent_json = json.dumps(entities)
                f.write(ent_json + "\n")

    def get_voc_entities(self) -> tuple:
        """
        Gets the texts, summaries, entites, words, and counts for the
        vocabulary.

        Returns:
            tuple: tuple of the texts, summaries, entities, words, and counts.
        """
        text = []
        summary = []
        entity = []
        allword = []
        cnt = 0

        with open(self.preprocessed_data_path, encoding="utf8") as f:
            for line in f:
                e = json.loads(line)

                # concatenate the sentences and section names
                if isinstance(e["text"], list) and isinstance(
                    e["text"][0], list
                ):
                    sents = catDoc(e["text"])
                    secs = catDoc(e["section_name"])
                    sents.extend(secs)
                else:
                    pass

                # concatenate the text and summary
                text = " ".join(sents)
                summary = " ".join(e["summary"])
                allword.extend(text.split())
                allword.extend(summary.split())

                # concatenate the entities
                entity.extend(getEnt(e["entity"]))
                cnt += 1
                if cnt % 2000 == 0:
                    print(cnt)

        return entity, allword

    def run_create_voc(self) -> None:
        """
        Runs the createVoc.py file from the SAPGraph model.
        Note that this code is heavily adapted from the original SAPGraph code.

        Credits:
            https://github.com/cece00/SAPGraph/blob/main/script/createVoc.py
        """
        # set up directories
        save_dir = os.path.join(CLEANED_DATA_PATH, self.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        saveFile = os.path.join(save_dir, "vocab")
        entFile = os.path.join(save_dir, "vocab_ent")

        entity, allword = self.get_voc_entities()

        fdist1 = nltk.FreqDist(allword)

        fout = open(saveFile, "w")
        keys = fdist1.most_common()
        for (
            key,
            val,
        ) in keys:  # key is the word while value is the frequency (times)
            try:
                fout.write("%s\t%d\n" % (key, val))
            except UnicodeEncodeError:
                continue
        fout.close()

        # write entities into file, to make entities in vocab
        fout_ent = open(entFile, "w")
        fdist2 = nltk.FreqDist(entity)
        keys2 = fdist2.most_common()
        k_left = 0
        for (
            key,
            val,
        ) in keys2:  # key is the word while value is the frequency (times)
            try:
                pass_sig = False
                for item in PUNCTUATION_FILTER:
                    if item in repr(key):
                        pass_sig = True
                        break
                if not pass_sig:
                    fout_ent.write("%s\t%s\t%d\n" % (key, key.lower(), val))
                    k_left += 1
            except UnicodeEncodeError:
                continue
        fout_ent.close()

    def run_low_tfidf_words(self) -> None:
        """
        Runs the lowTfidfWords.py file from the SAPGraph model.
        Note that this code is heavily adapted from the original SAPGraph code.

        Credits:
            https://github.com/cece00/SAPGraph/blob/main/script/lowTFIDFWords.py
        """
        save_dir = os.path.join(CLEANED_DATA_PATH, self.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saveFile = os.path.join(save_dir, "filter_word.txt")

        documents = []
        with open(self.preprocessed_data_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if isinstance(e["text"], list) and isinstance(
                    e["text"][0], list
                ):
                    text = catDoc(e["text"])
                else:
                    text = e["text"]
                documents.append(" ".join(text))

        vectorizer, tfidf_matrix = calTFidf(documents)
        print(
            "The number of example is %d, and the TFIDF vocabulary size is %d"
            % (len(documents), len(vectorizer.vocabulary_))
        )
        word_tfidf = np.array(tfidf_matrix.mean(0))
        del tfidf_matrix
        word_order = np.argsort(word_tfidf[0])  # sort A->Z, return index

        id2word = vectorizer.get_feature_names_out()
        with open(saveFile, "w") as fout:
            for idx in word_order:
                w = id2word[idx]
                fout.write(w + "\n")

    def run_cal_edge(self) -> None:
        """
        Runs the calEdge.py file from the SAPGraph model.
        Note that this code is heavily adapted from the original SAPGraph code.

        Credits:
            https://github.com/cece00/SAPGraph/blob/main/script/calEdge.py
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data_path",
            type=str,
            default=self.preprocessed_data_path,
            help="File to deal with",
        )
        parser.add_argument(
            "--dataset", type=str, default=self.dataset, help="dataset name"
        )
        args = parser.parse_args()

        if self.dev_environment:
            main_train(args)
        else:
            main(args)

    def preprocess(self) -> None:
        """
        Preprocesses the data for in preparation for the SAPGraph summarization.
        """
        print("Preprocessing data...")

        # grab only the rows where the text and abstract are not null
        print("\tSubsetting data...")
        self.subset_df.dropna(inplace=True)
        self.subset_df = self.subset_df.loc[
            (self.subset_df.loc[:, "text"].apply(len) > 0)
            & self.subset_df.loc[:, "abstract"].apply(len)
            > 0,
            :,
        ]

        self.subset_df.reset_index(drop=True, inplace=True)

        # extract entities
        print("\tGetting text entities...")
        self.subset_df["text_entities"] = self.subset_df.loc[
            :, ["text", "abstract"]
        ].apply(self.extract_text_components, axis=1)
        self.subset_df.dropna(inplace=True)

        # save the data to a json file
        print("\tSaving data to json file...")
        self.save_json()

        # create the vocabulary
        print("\tCreating vocabulary...")
        self.run_create_voc()

        # run low tfidf words
        print("\tRunning low tfidf words...")
        self.run_low_tfidf_words()

        # run cal edge
        print("\tRunning cal edge...")
        self.run_cal_edge()

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
