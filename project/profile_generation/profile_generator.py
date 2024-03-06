#!/usr/bin/env python3
'''
Generates summaries of articles based on categories of interest.

Authors: Kate Habich, Jen Yeaton
'''
import project.data
import numpy as np
from project.utils.functions import load_file_to_df
from project.topic_modeling.topmod_weights2df import Topic_Model_Class
from project.journal_summarization.LSTM_summarizer import LSTMSummarizer


class ProfileGenerator:

    def __init__(self, num_articles):
        self.num_journals = num_articles
        self.moral_foundations = self._collect_moral_foundations()
        self.topics = self._collect_topics()


    def _collect_topics(self):
        '''
        Collects all 10 topics and returns df of lists of words comprising 
        each topic.
        '''
        model_weights_file_path = 'project/topic_modeling/reddit_all_comments_10_topmod'
        topic_model = Topic_Model_Class(model_weights_file_path)
        topics_df = topic_model.weights2df()
        topic_words_column = topics_df.Representation
        return topic_words_column

    def _collect_moral_foundations(self):
        return 'moral foundation 1'

    def _collect_keywords(self, article_data):
        '''
        Collects keywords from journal articles and returns df column 
        containing list of keyword terms for each article.
        '''
        keywords = article_data['key_terms']
        return keywords

    def _create_topic_vector(self):
        return topic_vec

    def _create_keyword_vector(self):
        return keyword_vec

    def _calc_cosine_similarity(self):
        '''
        Calcultes cosine similarity for two individual vectors.
        '''
        topic_vec = self._create_topic_vector()
        keyword_vec = self._create_keyword_vector()
        sim = np.dot(topic_vec, keyword_vec) / \
            (np.linalg.norm(topic_vec) * np.linalg.norm(keyword_vec))
        return sim

    def _select_articles(self):
        # pass in self.num_articles
        pass

    def _make_long_summary(self):
        summarizer = LSTMSummarizer(dev_environment=False, 
                                    load_presaved_model=True)

    def _make_short_summary(self):

        pass


    def generate_profile(self):
        article_data = load_file_to_df('project/data/proquest_data_cleaned.fea')
        keywords = self._collect_keywords(article_data)
        return keywords


    # def __repr__(self):
    #     s = f"Moral Foundations: {self.moral_foundations}\n \
    #     Topics: {self.topics}\n \
    #     Number of articles: {self.num_journals}"
        
    #     return s
    

generator = ProfileGenerator(3)
profile = generator.generate_profile()
print(profile)