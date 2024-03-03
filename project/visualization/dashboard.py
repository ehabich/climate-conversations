# Import necessary libraries
import os
import sys

import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_collection.moral_foundation_data import (
    load_comments,
    load_submissions,
)


## IMPORT DATA
comment_df = load_comments()
submission_df = load_submissions()

## DOMINANT MORAL FOUNDATION HEATMAP
subreddits = [
    "climate",
    "climateskeptics",
    "climatechange",
    "environment",
    "climateoffensive",
    "science",
]
results_df = pd.DataFrame()
for sub in subreddits:
    subreddit = comment_df[comment_df["subreddit"] == sub]
    counts = subreddit.groupby(["Dominant_Moral_Foundation_Agg"])["id"].count()
    total = counts.sum()
    percentages = round((counts / total) * 100, 2)
    percentages_df = percentages.reset_index()
    percentages_df["subreddit"] = sub
    results_df = pd.concat([results_df, percentages_df], ignore_index=True)

results_df.columns = [
    "Dominant_Moral_Foundation_Agg",
    "Percentage",
    "Subreddit",
]

final_table = results_df.pivot(
    index="Dominant_Moral_Foundation_Agg",
    columns="Subreddit",
    values="Percentage",
)
print("final_tab", final_table.columns)
Dominant_Moral_Foundations_Agg = final_table.index.tolist()

# Plotly heatmap
fig = px.imshow(
    final_table,
    labels={
        "x": "Subreddit",
        "y": "Dominant Moral Foundation",
        "color": "Percentage",
    },
    x=subreddits,
    y=Dominant_Moral_Foundations_Agg,
    aspect="auto",
    title="Dominant Moral Foundations by Subreddit",
)


# Initialize the Dash app (ensure to use dash.Dash, not dashboard.Dash based on your code snippet)
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(
    children=[
        html.H1(children="Moral Foundations Visualization"),
        html.Div(
            children="""
        Visualization of dominant moral foundations across various subreddits.
    """
        ),
        # Include the Plotly heatmap
        dcc.Graph(id="moral-foundations-heatmap", figure=fig),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
