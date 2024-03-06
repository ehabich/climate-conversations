import os


# base paths
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "cleaned")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")
TOKENIZED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "tokenized")
MODEL_PATH = os.path.join(PROJECT_PATH, "data", "models")

# files
CLEANED_PROQUEST_FILE = os.path.join(
    CLEANED_DATA_PATH, "proquest", "ES-journals_cleaned.parquet"
)

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

# section headers
SECTION_HEADERS_MAPPING = {
    "Introduction": ["Introduction", "Purpose", "Background"],
    "Method": [
        "Literature Review",
        "Related Works",
        "Methodology",
        "Methods",
        "Method",
        "Method and Data",
        "Design and Methodology",
        "Design",
        "Materials and Methods",
        "Experimental Design",
        "Data Collection",
        "Data Analysis",
        "Experimental Section",
    ],
    "Result": [
        "Result",
        "Results",
        "Finding",
        "Findings",
        "Discussion",
        "Results and Discussion",
    ],
    "Conclusion": [
        "Conclusion",
        "Conclusions",
        "Conclusion and Future Work",
        "Conclusion and recommendation",
        "Implications",
        "Future Directions",
        "Recommendations for Policymakers",
        "Summary and discussion",
        "Future Directions",
        "Implications",
        "Policy Implications",
        "Limitations",
        "Conclusions and future perspectives",
    ],
    "Other": [
        "Funding",
        "Ethics Statement",
        "Acknowledgments",
        "Conflict of interest",
        "Publisher's Note",
        "Supplementary Material",
        "References",
        "Author contributions",
        "Authors’ Contributions",
        "Footnotes Disclaimer/Publisher’s Note",
    ],
}

PUNCTUATION_FILTER = [
    ",",
    ":",
    ";",
    "?",
    "&",
    "!",
    "*",
    "@",
    "$",
    "%",
    "\\",
    "`",
    "``",
    "|",
    "/",
]
