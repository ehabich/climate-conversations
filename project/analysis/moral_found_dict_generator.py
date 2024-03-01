"""
Expands a moral foundations dictionary using word embeddings.

Loads a moral foundations dictionary,  expands it by finding similar words using
a specified word embedding model, and then exports the expanded dictionary
to a JSON file.The expansion is based on a similarity threshold.

Author(s): Kathryn Link-Oberstar
"""

import json
import os
from gensim.models import KeyedVectors


def load_and_expand_moral_foundations_dictionary(
    dic_file_path,
    embedding_model="glove-twitter-200",
    similarity_threshold=0.65,
    num_words_to_expand=100,
):
    """
    Loads & expands a moral foundations dictionary using a word embedding model.
    """
    if os.path.exists("wordvectors.kv"):
        model = KeyedVectors.load("wordvectors.kv")
    else:
        import gensim.downloader as api
        model = api.load(embedding_model)
        model.save("wordvectors.kv")

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
    #word_to_moral_foundation_expanded_swap = swap_keys_values(word_to_moral_foundation_expanded)
    #print(word_to_moral_foundation_expanded_swap)
    expanded_dictionary = {}

    for word, categories in word_to_moral_foundation.items():
        if word in model.key_to_index:
            print('word in model', word, categories)
            similar_words = model.most_similar(
                positive=[word], topn=num_words_to_expand
            )

            for similar_word, similarity_score in similar_words:
                if similarity_score >= similarity_threshold:
                    print(similar_word, similarity_score)
                    if similar_word in expanded_dictionary.keys():
                        pass 
                    else:
                        expanded_dictionary[similar_word] = []
                    for cat in categories:
                            if cat not in expanded_dictionary[similar_word]:
                                expanded_dictionary[similar_word].append(cat)

    
    word_to_moral_foundation_expanded.update(expanded_dictionary)
    
    return word_to_moral_foundation_expanded

def swap_keys_values(d):
    swapped = {}
    for key, value_list in d.items():
        for value in value_list:
            if value in swapped:
                if len(swapped[value]) < 30:
                    swapped[value].append(key)
                else:
                    pass
            else:
                swapped[value] = [key]
    return swapped

def main():
    dic_file_path = "moral foundations dictionary.dic"
    embedding_model = "glove-twitter-200"
    similarity_threshold = 0.62
    num_words_to_expand = 100

    expanded_dictionary = load_and_expand_moral_foundations_dictionary(
        dic_file_path,
        embedding_model,
        similarity_threshold,
        num_words_to_expand,
    )

    expanded_dictionary = swap_keys_values(expanded_dictionary)

    del expanded_dictionary['MoralityGeneral']

    with open(
        "expanded_moral_foundations_dictionary.json", "w", encoding="utf-8"
    ) as json_file:
        json.dump(expanded_dictionary, json_file, ensure_ascii=False, indent=4)

    tot_words = 0
    for key, val in expanded_dictionary.items():
        print(f'{len(val)} words loaded for {key}.')
        tot_words += len(val)

    print(
        "Expanded dictionary loaded with",
        tot_words,
        "entries and dumped to JSON.",
    )

if __name__ == "__main__":
    main()
