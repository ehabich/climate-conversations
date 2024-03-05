import os
import sys
import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from io import BytesIO  
import base64
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_collection.moral_foundation_data import load_comments, load_submissions

comment_df = load_comments()
submission_df = load_submissions()

subreddits = [
    "climate", "climateskeptics", "climatechange",
    "environment", "climateoffensive", "science",
    "politics", "worldnews"
]

# Calculate percentage of dominant moral foundations for each subreddit
results_df = pd.DataFrame()

for sub in subreddits:
    subreddit_data = comment_df[comment_df["subreddit"] == sub]
    counts = subreddit_data.groupby(["Dominant_Moral_Foundation_Agg", "Hurricane_Period"])["id"].count().reset_index(name='counts')
    total_counts = counts.groupby("Hurricane_Period")["counts"].transform('sum')
    counts['Percentage'] = (counts['counts'] / total_counts) * 100
    counts["subreddit"] = sub
    results_df = pd.concat([results_df, counts], ignore_index=True)

final_table = results_df.pivot_table(
    index="Dominant_Moral_Foundation_Agg",
    columns="subreddit",
    values="Percentage",
    aggfunc='mean' 
).reset_index()

app = dash.Dash(__name__)

# Define the layout of the app with components for visualization
app.layout = html.Div([
    html.H1("Climate Conversations: Moral Foundations Across Climate Related Subreddits", style={'font-family': 'sans-serif'}),
    html.Div("Visualization of dominant moral foundations across various subreddits related to climate.", style={'font-family': 'sans-serif'}),
    html.Div([
        dcc.Dropdown(
            id='subreddit-selector',
            options=[{'label': 'All Subreddits', 'value': 'All'}] + [{'label': sub, 'value': sub} for sub in subreddits],
            value='All', 
            style={'width': '50%', 'display': 'inline-block', 'font-family': 'sans-serif'}
        ),
        dcc.Dropdown(
            id='hurricane-period-selector',
            options=[
                {'label': 'All Periods', 'value': 'All'},
                {'label': 'Before Hurricane', 'value': 'Before Hurricane'},
                {'label': 'During Hurricane', 'value': 'During Hurricane'},
                {'label': 'After Hurricane', 'value': 'After Hurricane'}
            ],
            value='All',
            style={'width': '50%', 'display': 'inline-block', 'font-family': 'sans-serif'}
        )
    ], style={'display': 'flex', 'font-family': 'sans-serif'}),
    html.Div([
        dcc.Graph(id="moral-foundations-heatmap", style={'width': '50%', 'display': 'inline-block'}),
        html.Img(id='word-cloud-img', style={'width': '50%', 'display': 'inline-block'}),
    ], style={'display': 'flex'}),
], style={'font-family': 'sans-serif', 'margin': '20px', 'background-color': '#f0f0f0'})

# Callback for updating bar chart
@app.callback(
    Output("moral-foundations-heatmap", "figure"),
    [Input('subreddit-selector', 'value'),
     Input('hurricane-period-selector', 'value')]
)
def update_heatmap(selected_subreddit, selected_hurricane_period):
    df = results_df.copy()

    if selected_hurricane_period == 'All':
        if selected_subreddit != 'All':
            df = df[df['subreddit'] == selected_subreddit]
            df = df.groupby(['Dominant_Moral_Foundation_Agg', 'subreddit'])['Percentage'].mean().reset_index()
        else:
            df = df.groupby(['Dominant_Moral_Foundation_Agg', 'subreddit'])['Percentage'].mean().reset_index()
    else:
        df = df[df['Hurricane_Period'] == selected_hurricane_period]
        if selected_subreddit != 'All':
            df = df[df['subreddit'] == selected_subreddit]

    fig = px.bar(
        df, 
        x="Dominant_Moral_Foundation_Agg", 
        y="Percentage", 
        color="subreddit", 
        title=f"Dominant Moral Foundations in {selected_subreddit if selected_subreddit != 'All' else 'all subreddits'} during {selected_hurricane_period if selected_hurricane_period != 'All' else 'all periods'}",
        barmode="group",
        category_orders={"Dominant_Moral_Foundation_Agg": df.groupby("Dominant_Moral_Foundation_Agg")["Percentage"].mean().sort_values(ascending=False).index}
    )
    return fig

def create_word_cloud_image(words_counter):
    word_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words_counter)
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    base64_img = base64.b64encode(img_bytes.read()).decode('utf8')
    plt.close()
    return 'data:image/png;base64,{}'.format(base64_img)

@app.callback(
    Output('word-cloud-img', 'src'),
    [Input('subreddit-selector', 'value'),
     Input('hurricane-period-selector', 'value')]
)
def update_word_cloud(selected_subreddit, selected_hurricane_period):
    df = comment_df.copy()
    if selected_hurricane_period != 'All':
        df = df[df['Hurricane_Period'] == selected_hurricane_period]

    if selected_subreddit != 'All':
        df = df[df['subreddit'] == selected_subreddit]

    excluded_words = {'remove', 'delete'}
    words_counter = Counter()

    for comment_words in df['tokenized_body_words_norm']:
        filtered_words = [word for word in comment_words if word not in excluded_words]
        words_counter.update(filtered_words)
    
    word_cloud_img = create_word_cloud_image(words_counter)
    return word_cloud_img

if __name__ == "__main__":
    app.run_server(debug=True)
