"""
Collects climate research articles for model development.
Uses search results generated from the Web of Science and scrapes research article text from source URLs.
Search results can be exported using this link: https://www.webofscience.com/wos/woscc/summary/b582483b-6474-4e85-a5e3-04609bbcbf5a-cbea852a/relevance/1(overlay:export/exc)

Author(s): Chanteria Milner
"""

# imports
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from time import sleep

# constants
from utils.constants import COLS_TO_KEEP, RENAMED_COLS, CHROME_DRIVER

# functions
from utils.functions import save


class ArticleScraper:
    """
    Extracts article text from source URL.

    parameters:
        filepath (str): path to file containing search results
    """

    def __init__(
        self,
        filepath: str,
        savepath: str = "climate-conversations/data/raw/climate_articles.csv",
    ):
        self.filepath = filepath
        self.savepath = savepath
        self.service = Service(executable_path=CHROME_DRIVER)
        self.driver = webdriver.Chrome(service=self.service)

        if any([filepath.endswith(".xls"), filepath.endswith(".xlsx")]):
            self.df = pd.read_excel(filepath)
        else:
            self.df = pd.read_excel(filepath)

        self.processed_df = pd.DataFrame(columns=RENAMED_COLS)

    def extract_text(self, doi: str):
        """
        Extracts article text from source URL.

        parameters:
            doi (str): article DOI
        """
        url = f"https://dx.doi.org/{doi}"

        # get page content
        self.driver.get(url)
        sleep(5)
        page = self.driver.page_source
        soup = BeautifulSoup(page, "html.parser")

        article_div_id = 

        # extract text
        text = soup.get_text()
        return text

    def process(self, verbose=True):
        """
        Iterates through search results and extracts the text of
        each article. Saves the results to a new csv.
        """
        for i, row in self.df.iterrows():
            if i % 50 == 0 and verbose:
                print(f"Processing article {i} of {len(self.df)}")
            try:
                url = row["DOI"]
                text = self.extract_text(url)
                data = row[COLS_TO_KEEP] + [text]
                self.processed_df = self.processed_df.append(
                    pd.Series(data, index=RENAMED_COLS), ignore_index=True
                )
            except:
                print(f"Error processing article {i}")
        save(self.processed_df, self.savepath)
        self.driver.quit()


def main(
    filepath="project/data/raw/WoS_results.xls",
    savepath="project/data/raw/climate_articles.csv",
    verbose=True,
):
    """
    Iterates through search results and extracts the text of
    each bill. Saves the results to a new csv.
    """
    # read in search results
    scraper = ArticleScraper(filepath=filepath, savepath=savepath)
    scraper.process(verbose=verbose)

    return True
