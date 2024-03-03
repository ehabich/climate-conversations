import os
import sys
import pandas as pd

def load_comments():
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
    directory = os.path.join(parent_directory,'data_collection/project_data/results/comments/')
    all_comments_tokenized_analyzed = directory_importer(directory)
    all_comments_tokenized_analyzed = df_processor(all_comments_tokenized_analyzed)
    return all_comments_tokenized_analyzed 

def load_submissions():
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
    directory = os.path.join(parent_directory,'data_collection/project_data/results/submissions/')
    all_submissions_tokenized_analyzed = directory_importer(directory)
    all_submissions_tokenized_analyzed = df_processor(all_submissions_tokenized_analyzed)
    return all_submissions_tokenized_analyzed 

def directory_importer(directory):
    final_df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            df = pd.read_pickle(file_path)
            final_df = pd.concat([final_df, df], ignore_index=True)
    return final_df

def df_processor(full_df):
    full_df['Harm_Care_Agg'] = (full_df['HarmVice'] + full_df['HarmVirtue']) / 2
    full_df['Authority_Agg'] = (full_df['AuthorityVice'] + full_df['HarmVirtue']) / 2
    full_df['Purity_Agg'] = (full_df['PurityVice'] + full_df['PurityVirtue']) / 2
    full_df['Fairness_Agg'] = (full_df['FairnessVice'] + full_df['FairnessVirtue']) / 2
    full_df['Ingroup_Agg'] = (full_df['IngroupVice'] + full_df['IngroupVirtue']) / 2
    full_df['Dominant_Moral_Foundation'] = full_df[['HarmVirtue', 'AuthorityVirtue', 'PurityVirtue', 'HarmVice', 'PurityVice', 'IngroupVice', 'FairnessVirtue', 'FairnessVice', 'IngroupVirtue', 'AuthorityVice']].idxmax(axis=1)
    full_df['Dominant_Moral_Foundation_Agg'] = full_df[['Harm_Care_Agg','Authority_Agg', 'Purity_Agg', 'Fairness_Agg', 'Ingroup_Agg']].idxmax(axis=1)
    return full_df

def main():
    comments_df = load_comments()
    submissions_df = load_submissions()

    print("Comments DataFrame:")
    print(comments_df.head())

    print("\nSubmissions DataFrame:")
    print(submissions_df.head())

if __name__ == "__main__":
    main()