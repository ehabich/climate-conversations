import pandas as pd


def load_file_to_df(filepath: str) -> pd.DataFrame:
    """
    Loads a file into a dataframe.

    parameters:
        filepath (str): path to the file to load.
        load_tokenized (bool): whether or not to load tokenized data.
        tokenized_cols (list): list of columns to load tokenized data for.

    returns:
        df (pd.DataFrame): dataframe of the file.
    """
    ext = filepath.split(".")[-1].lower()

    if ext in ["pickle", "pkl"]:
        df = pd.read_pickle(filepath)
    elif ext in ["csv", "txt"]:
        df = pd.read_csv(filepath)
    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(filepath)
    elif ext in ["fea", "feather"]:
        df = pd.read_feather(filepath)
    elif ext in ["parquet", "pq"]:
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"File type {ext} not supported.")

    return df


def save(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the given dataframe out to the specified filepath.

    parameters:
        df (pd.DataFrame): dataframe to save.
        filepath (str): path to save the dataframe to.
    """
    ext = filepath.split(".")[-1]

    if ext in ["pickle", "pkl"]:
        df.to_pickle(filepath)
    elif ext in ["csv", "txt"]:
        df.to_csv(filepath, index=False)
    elif ext in ["xlsx", "xls"]:
        df.to_excel(filepath, index=False)
    elif ext in ["fea", "feather"]:
        df.to_feather(filepath)
    elif ext in ["parquet", "pq"]:
        df.to_parquet(filepath)
    else:
        raise ValueError(f"File type {ext} not supported.")
