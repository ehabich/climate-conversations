#!/usr/bin/env python3
'''
Preprocess cleaned journals for abstractive summarization model.

Author: Kate Habich
'''
from project.utils.functions import load_file_to_df
from datasets import Dataset
import pandas as pd

def preprocess_journal_data():
    # Read in data
    data = pd.read_feather('project/data/proquest_data_cleaned.fea')

    # Select relevant columns and rows
    limited_data = data.loc[(data['title'].notnull()) & 
                            (data['abstract'].notnull()) ,
                            ['title', 'abstract']].rename(
        columns = {'abstract': 'text', 'title':'summary'})
  
    # Cast df to Dataset type for HF model
    preprocessed_data = Dataset.from_pandas(limited_data)

    return preprocessed_data

preprocess_journal_data()