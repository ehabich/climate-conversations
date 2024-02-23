"""
Creates a data cleaner class to clean the proquest data.

Author(s): Chanteria Milner
"""

import json
import re

import pandas as pd
from bs4 import BeautifulSoup

from project.utils.constants import ARTICLE_DATA_COLUMNS
from project.utils.functions import load_file_to_df


class ProquestCleaner:
    """
    Cleans data exported from Proquest TDM.

    Args:
        file_path (str): path to the file to clean.
    """

    def __init__(self, file_path):
        self.df = load_file_to_df(file_path)
        self.cleaned_df = None

    def extract_article_text(self, text: str) -> str:
        """
        Extracts article text from the Proquest data.

        Args:
            text (str): text to extract article text from.

        Returns:
            str: extracted article text.
        """
        data = json.loads(text)["RECORD"]

        if data.get("TextInfo"):
            if data["TextInfo"].get("Text"):
                text_xml = data["TextInfo"]["Text"]["#text"]
            elif data["TextInfo"].get("HiddenText"):
                text_xml = data["TextInfo"]["HiddenText"]
            else:
                text_xml = None

            if text_xml:
                text_soup = BeautifulSoup(text_xml, "lxml")
                text = self.clean_article_text(text_soup.find_all("p"))

                return text

        return None

    def extract_article_abstract(self, text: str) -> str:
        """
        Extracts article abstract from the Proquest data.

        Args:
            text (str): text to extract article abstract from.

        Returns:
            str: extracted article abstract.
        """
        data = json.loads(text)["RECORD"]

        if data.get("Obj"):
            if data["Obj"].get("Abstract"):
                abstracts = data["Obj"]["Abstract"]
                if isinstance(abstracts, list):
                    abstract_xml = abstracts[0]["Medium"]["AbsText"]["#text"]
                else:
                    abstract_xml = data["Obj"]["Abstract"]["Medium"]["AbsText"][
                        "#text"
                    ]
                abstract_soup = BeautifulSoup(abstract_xml, "lxml")
                abstract = self.clean_article_text(abstract_soup.find_all("p"))

                return abstract

        return None

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans text.

        Args:
            text (str): text to clean.

        Returns:
            str: cleaned text.
        """
        cleaned_text = re.sub(r"<.*?>|\n", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text

    def clean_article_text(self, paragraphs: list) -> str:
        """
        Cleans text article paragraphs and converts it into a single string.

        Args:
            paragraphs (list): text to clean.

        Returns:
            str: cleaned text.
        """
        clean_text = ""

        for paragraph in paragraphs:
            cleaned_p = str(paragraph)
            clean_text += self.clean_text(cleaned_p) + " "

        return clean_text

    @staticmethod
    def extract_authors(contributors: list) -> list:
        """
        Extracts authors from the Proquest data.

        Args:
            contributors (list): list of contributors to extract authors from.

        Returns:
            list: extracted authors.
        """
        if isinstance(contributors, dict):
            if contributors.get("Author"):
                author = {
                    "first_name": "",
                    "last_name": "",
                }
                if contributors["Author"].get("FirstNameAtt"):
                    author["first_name"] = contributors["Author"][
                        "FirstNameAtt"
                    ]["FirstName"]

                if contributors["Author"].get("LastNameAtt"):
                    author["last_name"] = contributors["Author"]["LastNameAtt"][
                        "LastName"
                    ]

                return [author]

        authors = []
        for contributor in contributors:
            if contributor.get("Author"):
                author = {
                    "first_name": "",
                    "last_name": "",
                }
                if contributor["Author"].get("FirstNameAtt"):
                    author["first_name"] = contributor["Author"][
                        "FirstNameAtt"
                    ]["FirstName"]

                if contributor["Author"].get("LastNameAtt"):
                    author["last_name"] = contributor["Author"]["LastNameAtt"][
                        "LastName"
                    ]
                authors.append(author)

        return authors

    def extract_key_terms(self, terms_json: json) -> list:
        """
        Extracts key terms from the Proquest data.

        Args:
            terms (json): object to draw terms from.

        Returns:
            list: extracted key terms.
        """
        key_terms = set()

        for _, terms in terms_json.items():
            if isinstance(terms, dict):
                to_add = list(terms.values())[-1]
                if isinstance(to_add, str):
                    to_add = self.clean_text(to_add).lower()
                    key_terms.add(to_add)
                continue
            for term in terms:
                to_add = list(term.values())[-1]
                if isinstance(to_add, str):
                    to_add = self.clean_text(to_add).lower()
                    key_terms.add(to_add)

        return ", ".join(list(key_terms))

    def extract_artcle_attrs(self, data: json) -> dict:
        """
        Extracts article attributes from the Proquest data.

        Args:
            data (json): data to extract attributes from.
            data_dict (dict): dictionary to store extracted attributes.
        """
        data_dict = {
            "title": (
                self.clean_text(data["Obj"]["TitleAtt"]["Title"])
                if data["Obj"].get("TitleAtt")
                else None
            ),
            "publication_date": (
                data["Obj"]["NumericDate"]
                if data["Obj"].get("NumericDate")
                else None
            ),
            "num_pages": (
                int(data["Obj"]["PageCount"])
                if data["Obj"].get("PageCount")
                else None
            ),
            "pagination": (
                data["Obj"]["PrintLocation"]["Pagination"]
                if data["Obj"].get("PrintLocation").get("Pagination")
                else None
            ),
            "doi": (
                data["Obj"]["ObjectIDs"]["ObjectID"][0]["DOI"]
                if data["Obj"]["ObjectIDs"]["ObjectID"][0].get("DOI")
                else None
            ),
            "publisher": (
                data["DFS"]["PubFrosting"]["publisher"]["PublisherName"]
                if data["DFS"]["PubFrosting"].get("publisher")
                else None
            ),
            "publication_title": (
                data["DFS"]["PubFrosting"]["Title"]
                if data["DFS"]["PubFrosting"].get("Title")
                else None
            ),
            "volume": (
                int(data["DFS"]["GroupFrosting"]["Volume"])
                if data["DFS"]["GroupFrosting"].get("Volume")
                else None
            ),
            "issue": (
                int(data["DFS"]["GroupFrosting"]["Issue"])
                if data["DFS"]["GroupFrosting"].get("Issue")
                else None
            ),
        }

        return data_dict

    def extract_info(self, text: str) -> str:
        """
        Extracts information from the Proquest data,
        including article attributes and article text.

        Args:
            text (str): information to extract.

        Returns:
            str: extracted information.
        """
        data_dict = {}
        for col in ARTICLE_DATA_COLUMNS:
            data_dict[col] = None

        data = json.loads(text)["RECORD"]

        # extract and clean article text
        data_dict["article_text"] = self.extract_article_text(text)

        if data.get("Obj"):
            # extract abstract
            data_dict["abstract"] = self.extract_article_abstract(text)

            # extract authors
            if data["Obj"].get("Contributors"):
                contributors = data["Obj"]["Contributors"]["Contributor"]
                authors = self.extract_authors(contributors)
                data_dict["authors"] = authors

            # extract key terms
            if data["Obj"].get("Terms"):
                terms_json = data["Obj"]["Terms"]
                key_terms = self.extract_key_terms(terms_json)
                data_dict["key_terms"] = key_terms

            # extract article attributes
            if data.get("DFS"):
                data_dict = data_dict | self.extract_artcle_attrs(data)

        return data_dict

    def clean(self) -> None:
        """
        Cleans the data.
        """
        cleaned_dict = {}
        for col in ARTICLE_DATA_COLUMNS:
            cleaned_dict[col] = []

        for idx in self.df.index:
            row = self.df.loc[idx]
            print(f"Cleaning row {idx} of {len(self.df)}...")
            article_info = self.extract_info(row["Data"])
            for key, value in article_info.items():
                cleaned_dict[key].append(value)

        self.cleaned_df = pd.DataFrame(cleaned_dict)
