import os


# base paths
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "cleaned")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")
TOKENIZED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "tokenized")

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
    "Introduction": "Introduction",
    "Purpose": "Introduction",
    "Background": "Introduction",
    "Literature Review": "Other",
    "Literature review": "Other",
    "Related Works": "Other",
    "Related Work": "Other",
    "Methodology": "Method",
    "Methods": "Method",
    "Method": "Method",
    "Design and Methodology": "Method",
    "Design": "Method",
    "Materials and methods": "Method",
    "Materials and Methods": "Method",
    "Experimental Design": "Method",
    "Experimental design": "Method",
    "Data collection": "Method",
    "Data Collection": "Method",
    "Data Analysis": "Method",
    "Data analysis": "Method",
    "Data and Methodology": "Method",
    "Data and methodology": "Method",
    "Data collection and analysis": "Method",
    "Data Collection and Analysis": "Method",
    "Result": "Result",
    "Results": "Result",
    "Finding": "Result",
    "Findings": "Result",
    "Discussion": "Result",
    "Conclusion": "Conclusion",
    "Conclusion and Future Work": "Conclusion",
    "Conclusions": "Conclusion",
    "Conclusion and recommendation": "Conclusion",
    "Conclusion and Recommendation": "Conclusion",
    "Conclusion and recommendations": "Conclusion",
    "Conclusion and Recommendations": "Conclusion",
    "Future Directions": "Conclusion",
    "Future directions": "Conclusion",
    "Implications": "Conclusion",
    "Policy Implications": "Conclusion",
    "Policy implications": "Conclusion",
    "Limitations": "Conclusion",
    "Funding": "Other",
    "Ethics Statement": "Other",
    "Ethics statement": "Other",
    "Acknowledgments": "Other",
    "Acknowledgements": "Other",
    "Acknowledgment": "Other",
    "Acknowledgement": "Other",
    "Conflict of interest": "Other",
    "Conflist of Interest": "Other",
    "Publisher's Note": "Other",
    "Footnotes Disclaimer/Publisher’s Note": "Other",
    "Publisher's note": "Other",
    "Supplementary Material": "Other",
    "Supplementary material": "Other",
    "References": "Other",
    "Author contributions": "Other",
    "Author Contributions": "Other",
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
