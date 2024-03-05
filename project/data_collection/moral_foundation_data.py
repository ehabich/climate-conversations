"""
Loads Processed Data for Visualization

Author(s): Kathryn Link-Oberstar
"""
import os

import pandas as pd


HURRICANE_START = pd.to_datetime("2022-09-23")
HURRICANE_END = pd.to_datetime("2022-09-30")

SUBREDDITS = [
    "climate",
    "climateskeptics",
    "climatechange",
    "environment",
    "climateoffensive",
    "science",
    "politics",
    "worldnews",
]


def load_comments():
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    directory = os.path.join(
        parent_directory, "data_collection/project_data/results/comments/"
    )
    all_comments_tokenized_analyzed = directory_importer(directory)
    all_comments_tokenized_analyzed = df_processor(
        all_comments_tokenized_analyzed
    )
    return all_comments_tokenized_analyzed


def load_submissions():
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    directory = os.path.join(
        parent_directory, "data_collection/project_data/results/submissions/"
    )
    all_submissions_tokenized_analyzed = directory_importer(directory)
    all_submissions_tokenized_analyzed = df_processor(
        all_submissions_tokenized_analyzed
    )
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
    full_df["Harm_Care_Agg"] = (full_df["HarmVice"] + full_df["HarmVirtue"]) / 2
    full_df["Authority_Agg"] = (
        full_df["AuthorityVice"] + full_df["HarmVirtue"]
    ) / 2
    full_df["Purity_Agg"] = (
        full_df["PurityVice"] + full_df["PurityVirtue"]
    ) / 2
    full_df["Fairness_Agg"] = (
        full_df["FairnessVice"] + full_df["FairnessVirtue"]
    ) / 2
    full_df["Ingroup_Agg"] = (
        full_df["IngroupVice"] + full_df["IngroupVirtue"]
    ) / 2
    full_df["Dominant_Moral_Foundation"] = full_df[
        [
            "HarmVirtue",
            "AuthorityVirtue",
            "PurityVirtue",
            "HarmVice",
            "PurityVice",
            "IngroupVice",
            "FairnessVirtue",
            "FairnessVice",
            "IngroupVirtue",
            "AuthorityVice",
        ]
    ].idxmax(axis=1)
    full_df["Dominant_Moral_Foundation_Agg"] = full_df[
        [
            "Harm_Care_Agg",
            "Authority_Agg",
            "Purity_Agg",
            "Fairness_Agg",
            "Ingroup_Agg",
        ]
    ].idxmax(axis=1)

    full_df["created_utc"] = pd.to_datetime(
        full_df["created_utc"], unit="s"
    )  # Convert Unix timestamp to datetime
    full_df["Hurricane_Period"] = full_df.apply(categorize_period, axis=1)

    return full_df


def categorize_period(row):
    if row["created_utc"] < HURRICANE_START:
        return "Before Hurricane"
    elif HURRICANE_START <= row["created_utc"] <= HURRICANE_END:
        return "During Hurricane"
    else:
        return "After Hurricane"


def comment_results_processor(comment_df):
    results_df = pd.DataFrame()
    for sub in SUBREDDITS:
        subreddit_data = comment_df[comment_df["subreddit"] == sub]
        counts = (
            subreddit_data.groupby(
                ["Dominant_Moral_Foundation_Agg", "Hurricane_Period"]
            )["id"]
            .count()
            .reset_index(name="counts")
        )
        total_counts = counts.groupby("Hurricane_Period")["counts"].transform(
            "sum"
        )
        counts["Percentage"] = (counts["counts"] / total_counts) * 100
        counts["subreddit"] = sub
        results_df = pd.concat([results_df, counts], ignore_index=True)

    results_df.pivot_table(
        index="Dominant_Moral_Foundation_Agg",
        columns="subreddit",
        values="Percentage",
        aggfunc="mean",
    ).reset_index()

    return results_df


def main():
    comments_df = load_comments()
    submissions_df = load_submissions()
    results_df = comment_results_processor(comments_df)

    print("Comments DataFrame:")
    print(comments_df.head())

    print("\nSubmissions DataFrame:")
    print(submissions_df.head())

    print("\nResults DataFrame:")
    print(results_df.head())


if __name__ == "__main__":
    main()
