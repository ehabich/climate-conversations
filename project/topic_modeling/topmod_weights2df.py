"""
Topic Modeling: Creates a class for the topic model based on its weights. 

Author: Jennifer Yeaton
"""

from bertopic import BERTopic
import pandas as pd
import sys
import os

parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_directory)


class Topic_Model_Class:
    """
    Class representing a topic model.

    """

    def __init__(self, model_weights_file_path):
        self.model_weights_file_path = model_weights_file_path
        self.topic_model = None

    def weights2df(self):
        """
        Read in the weights of a topic model and return a dataframe.
        The dataframe will be used

        Input:
            model_weights_file()

        Returns:
            model_df_representation(df): Dataframe containing the model weights.
        """

        self.topic_model = BERTopic.load(self.model_weights_file_path)
        model_df_representation = topic_model.get_topic_info()

        return model_df_representation

    def visualize(self):
        """
        https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-topics-per-class

        """
        self.topic_model.visualize_barchart()
        self.topic_model.visualize_topics()
