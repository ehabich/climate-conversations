import os


# base paths
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "cleaned")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")
TOKENIZED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "tokenized")

# data attributes
ARTICLE_DATA_COLUMNS = [
    "title",
    "publication_date",
    "num_pages",
    "pagination",
    "doi",
    "publisher",
    "publication_title",
    "volume",
    "issue",
    "text",
    "abstract",
    "authors",
    "key_terms",
]
