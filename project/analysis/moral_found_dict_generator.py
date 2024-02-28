"""
Expands a moral foundations dictionary using word embeddings.

Loads a moral foundations dictionary,  expands it by finding similar words using
a specified word embedding model, and then exports the expanded dictionary
to a JSON file.The expansion is based on a similarity threshold.

Author(s): Kathryn Link-Oberstar
"""

import json

import gensim.downloader as api


def load_and_expand_moral_foundations_dictionary(
    dic_file_path,
    embedding_model="glove-twitter-25",
    similarity_threshold=0.85,
    num_words_to_expand=100,
):
    """
    Loads & expands a moral foundations dictionary using a word embedding model.
    """
    model = api.load(embedding_model)

    moral_foundations_dict = {}
    word_to_moral_foundation = {}

    with open(dic_file_path, "r", encoding="utf-8") as file:
        line_counter = 0
        for line in file:
            if line.strip() and not line.startswith("%"):
                parts = line.strip().split()
                moral_found = parts[0].rstrip("*")
                moral_found_code = parts[1:]
                if line_counter < 12:
                    moral_foundations_dict[moral_found] = moral_found_code[0]
                else:
                    cats = [
                        moral_foundations_dict.get(cat, cat)
                        for cat in moral_found_code
                    ]  # Ensure to get the category or keep the original
                    word_to_moral_foundation[moral_found] = cats
                line_counter += 1

    word_to_moral_foundation_expanded = word_to_moral_foundation.copy()
    expanded_dictionary = {}

    for word, categories in word_to_moral_foundation.items():
        if word in model.key_to_index:
            similar_words = model.most_similar(
                positive=[word], topn=num_words_to_expand
            )

            for similar_word, similarity_score in similar_words:
                if similarity_score >= similarity_threshold:
                    expanded_dictionary[similar_word] = categories

    word_to_moral_foundation_expanded.update(expanded_dictionary)

    return word_to_moral_foundation_expanded


def main():
    dic_file_path = "moral foundations dictionary.dic"
    embedding_model = "glove-twitter-25"
    similarity_threshold = 0.85
    num_words_to_expand = 100

    expanded_dictionary = load_and_expand_moral_foundations_dictionary(
        dic_file_path,
        embedding_model,
        similarity_threshold,
        num_words_to_expand,
    )

    with open(
        "expanded_moral_foundations_dictionary.json", "w", encoding="utf-8"
    ) as json_file:
        json.dump(expanded_dictionary, json_file, ensure_ascii=False, indent=4)

    print(
        "Expanded dictionary loaded with",
        len(expanded_dictionary),
        "entries and dumped to JSON.",
    )


if __name__ == "__main__":
    main()
