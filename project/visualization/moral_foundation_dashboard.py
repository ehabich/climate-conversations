"""
Create Dashboard to Visualize Moral Foundation Analysis of Reddit Comments.

Author(s): Kathryn Link-Oberstar
"""
import os
import sys
from itertools import cycle

import dash
import matplotlib
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from wordcloud import WordCloud


matplotlib.use("Agg")
import base64
from collections import Counter
from io import BytesIO

import matplotlib.pyplot as plt
from flask_caching import Cache


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_collection.moral_foundation_data import (
    comment_results_processor,
    load_comments,
    load_submissions,
)


# Load Data
comment_df = load_comments()
submission_df = load_submissions()
results_df = comment_results_processor(comment_df)

# Initialize subreddits and colors
subreddits = [
    "climate",
    "climateskeptics",
    "climatechange",
    "environment",
    "climateoffensive",
    "science",
    "politics",
    "worldnews",
]

COLORS = cycle(
    [
        "#c7522a",
        "#e5c185",
        "#f0daa5",
        "#fbf2c4",
        "#b8cdab",
        "#74a892",
        "#008585",
        "#004343",
    ]
)


def color_func(word, **kwargs):
    return next(COLORS)


# Initialize the dash app
app = dash.Dash(__name__)

# Set up caching
cache = Cache(
    app.server,
    config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"},
)

timeout = 100000000

# Specify layout for the app
app.layout = html.Div(
    [
        html.H1(
            "Climate Conversations: Dominant Moral Foundations Across Climate Related Subreddits",
            style={"font-family": "sans-serif"},
        ),
        html.Div(
            "Moral foundation theory argues that there are five basic moral foundations: (1) harm/care, (2) fairness/reciprocity, (3) ingroup/loyalty, (4) authority/respect, and (5) purity/sanctity. These five foundations comprise the building blocks of morality, regardless of the culture. The following analyzes data from 8 subreddits related to climate based on the dominant moral foundations. Major weather events tend to increase discussion of climate online. To inspect the effect of this increased discourse, we analyze reddit comments before, during, and after Hurricane Ian (September 23, 2022 – September 30, 2022).",
            style={
                "font-family": "sans-serif",
                "margin-bottom": "20px",
                "font-style": "italic",
            },
        ),
        # Dropdowns section
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Select a subreddit:",
                            style={"font-family": "sans-serif"},
                        ),
                        dcc.Dropdown(
                            id="subreddit-selector",
                            options=[
                                {"label": "All Subreddits", "value": "All"}
                            ]
                            + [
                                {"label": sub, "value": sub}
                                for sub in subreddits
                            ],
                            value="All",
                            style={
                                "width": "100%",
                                "font-family": "sans-serif",
                            },
                        ),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "margin-right": "2%",
                        "vertical-align": "top",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Select a time period:",
                            style={"font-family": "sans-serif"},
                        ),
                        dcc.Dropdown(
                            id="hurricane-period-selector",
                            options=[
                                {"label": "All Periods", "value": "All"},
                                {
                                    "label": "Before Hurricane",
                                    "value": "Before Hurricane",
                                },
                                {
                                    "label": "During Hurricane",
                                    "value": "During Hurricane",
                                },
                                {
                                    "label": "After Hurricane",
                                    "value": "After Hurricane",
                                },
                            ],
                            value="All",
                            style={
                                "width": "100%",
                                "font-family": "sans-serif",
                            },
                        ),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                ),
            ],
            style={"font-family": "sans-serif", "margin-bottom": "20px"},
        ),
        html.H3(
            "Frequency of Dominant Moral Foundation",
            style={"font-family": "sans-serif"},
        ),
        html.Div(
            "Comments were classified by their dominant moral foundation by taking the cosine similarity between each comment and words in an Expanded Moral Foundation Dictionary. Each comment is classified to the moral foundation with which it is most aligned. The percentages reflect the percent of comments in that subreddit classified to that moral foundation.",
            style={"font-family": "sans-serif"},
        ),
        dcc.Graph(id="moral-foundations-barchart"),
        html.H3(
            "Word Cloud with Most Common Words",
            style={"font-family": "sans-serif"},
        ),
        html.Div(
            [
                html.Img(
                    id="word-cloud-img",
                    style={
                        "max-width": "100%",
                        "height": "auto",
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                    },
                )
            ],
            style={"text-align": "center"},
        ),
    ],
    style={
        "font-family": "sans-serif",
        "margin": "20px",
        "background-color": "#FFFFFF",
    },
)


# App callback for the bar chart
@app.callback(
    Output("moral-foundations-barchart", "figure"),
    [
        Input("subreddit-selector", "value"),
        Input("hurricane-period-selector", "value"),
    ],
)

# Helper functions to create and update the bar charts
def update_barchart(selected_subreddit, selected_hurricane_period):
    df = results_df.copy()

    if selected_hurricane_period == "All":
        if selected_subreddit != "All":
            df = df[df["subreddit"] == selected_subreddit]
            df = (
                df.groupby(["Dominant_Moral_Foundation_Agg", "subreddit"])[
                    "Percentage"
                ]
                .mean()
                .reset_index()
            )
        else:
            df = (
                df.groupby(["Dominant_Moral_Foundation_Agg", "subreddit"])[
                    "Percentage"
                ]
                .mean()
                .reset_index()
            )
    else:
        df = df[df["Hurricane_Period"] == selected_hurricane_period]
        if selected_subreddit != "All":
            df = df[df["subreddit"] == selected_subreddit]

    mf_labels = {
        "Harm_Care_Agg": "Care & Harm",
        "Fairness_Agg": "Fairness & Reciprocity",
        "Authority_Agg": "Authority & Respect",
        "Purity_Agg": "Purity & Sanctity",
        "Ingroup_Agg": "Ingroup & Loyalty",
    }

    subreddit_labels = {
        "climate": "r/climate",
        "climateskeptics": "r/climateskeptics",
        "climatechange": "r/climatechange",
        "environment": "r/environment",
        "climateoffensive": "r/climateoffensive",
        "science": "r/science",
        "politics": "r/politics",
        "worldnews": "r/worldnews",
    }

    custom_colors = {
        "r/climate": "#c7522a",
        "r/climateskeptics": "#e5c185",
        "r/climatechange": "#f0daa5",
        "r/environment": "#fbf2c4",
        "r/climateoffensive": "#b8cdab",
        "r/science": "#74a892",
        "r/politics": "#008585",
        "r/worldnews": "#004343",
    }

    df["Dominant_Moral_Foundation_Agg"] = (
        df["Dominant_Moral_Foundation_Agg"]
        .map(mf_labels)
        .fillna(df["Dominant_Moral_Foundation_Agg"])
    )
    df["subreddit"] = (
        df["subreddit"].map(subreddit_labels).fillna(df["subreddit"])
    )

    fig = px.bar(
        df,
        x="Dominant_Moral_Foundation_Agg",
        y="Percentage",
        color="subreddit",
        title=f"Dominant Moral Foundations in {selected_subreddit if selected_subreddit != 'All' else 'all subreddits'} during {selected_hurricane_period if selected_hurricane_period != 'All' else 'all periods'}",
        barmode="group",
        category_orders={
            "Dominant_Moral_Foundation_Agg": df.groupby(
                "Dominant_Moral_Foundation_Agg"
            )["Percentage"]
            .mean()
            .sort_values(ascending=False)
            .index
        },
        color_discrete_map=custom_colors,
    )

    fig.update_layout(
        xaxis_title="Dominant Moral Foundation",
        yaxis_title="Percentage (%)",
        legend_title="Subreddit",
        yaxis={"range": [0, 100]},
    )

    return fig


# App callback for the word cloud
@app.callback(
    Output("word-cloud-img", "src"),
    [
        Input("subreddit-selector", "value"),
        Input("hurricane-period-selector", "value"),
    ],
)

# Helper functions to create the word clouds
def update_word_cloud(selected_subreddit, selected_hurricane_period):
    df = comment_df.copy()
    if selected_hurricane_period != "All":
        df = df[df["Hurricane_Period"] == selected_hurricane_period]

    if selected_subreddit != "All":
        df = df[df["subreddit"] == selected_subreddit]

    excluded_words = {"remove", "delete"}
    words_counter = Counter()

    for comment_words in df["tokenized_body_words_norm"]:
        filtered_words = [
            word for word in comment_words if word not in excluded_words
        ]
        words_counter.update(filtered_words)

    word_cloud = WordCloud(
        width=800, height=400, background_color="white", color_func=color_func
    ).generate_from_frequencies(words_counter)
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png")
    img_bytes.seek(0)
    base64_img = base64.b64encode(img_bytes.read()).decode("utf8")
    plt.close()
    return "data:image/png;base64,{}".format(base64_img)


if __name__ == "__main__":
    app.run_server(debug=True)
