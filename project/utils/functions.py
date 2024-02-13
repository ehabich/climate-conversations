"""
Common utility functions.
"""

import os
import re
import pandas as pd
import requests

def save(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the given dataframe out to the specified filepath.
    """
    df.to_csv(filepath, index=False)
