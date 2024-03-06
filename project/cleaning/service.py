"""
Service for cleaning data.
"""

import glob
import os

import pandas as pd

from project.cleaning.proquest_clean import ProquestCleaner
from project.utils.constants import CLEANED_DATA_PATH, RAW_DATA_PATH
from project.utils.functions import save_df_to_file


def concat_climate_articles() -> None:
    """
    Concatenates climate articles from Proquest data.
    """
    # grab data files
    proquest_files = glob.glob(
        os.path.join(RAW_DATA_PATH, "proquest", "ES-journals_*.parquet")
    )

    # load data
    proquest_dfs = [pd.read_parquet(file) for file in proquest_files]

    # concatenate data
    proquest_df = pd.concat(proquest_dfs, ignore_index=True)

    # save data
    save_df_to_file(
        proquest_df,
        os.path.join(RAW_DATA_PATH, "proquest", "ES-journals.parquet"),
    )


def run_proquest_clean() -> None:
    """
    Cleans data exported from Proquest TDM.
    """
    # ensure concatenated data file exists
    if not os.path.exists(
        os.path.join(RAW_DATA_PATH, "proquest", "ES-journals.parquet")
    ):
        concat_climate_articles()

    cleaner = ProquestCleaner(
        os.path.join(RAW_DATA_PATH, "proquest", "ES-journals.parquet")
    )

    cleaner.clean()
    if not os.path.exists(os.path.join(CLEANED_DATA_PATH, "proquest")):
        os.makedirs(os.path.join(CLEANED_DATA_PATH, "proquest"))
    save_df_to_file(
        cleaner.cleaned_df,
        os.path.join(
            CLEANED_DATA_PATH, "proquest", "ES-journals_cleaned.parquet"
        ),
    )
