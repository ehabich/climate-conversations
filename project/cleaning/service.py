"""
Service for cleaning data.
"""

import os

from project.cleaning.proquest_clean import ProquestClean
from project.utils.functions import save
from project.utils.constants import RAW_DATA_PATH, CLEANED_DATA_PATH


def run_proquest_clean() -> None:
    """
    Cleans data exported from Proquest TDM.
    """
    cleaner = ProquestClean(os.path.join(RAW_DATA_PATH, "proquest_data.parquet"))

    cleaner.clean()
    save(
        cleaner.cleaned_df,
        os.path.join(CLEANED_DATA_PATH, "proquest_data_cleaned.parquet"),
    )
