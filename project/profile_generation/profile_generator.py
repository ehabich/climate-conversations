#!/usr/bin/env python3
'''
Generates summaries of articles based on categories of interest.

Authors: Kate Habich, Jen Yeaton
'''
import os
import project.data
import numpy as np
import pandas as pd
from functools import partial
# from project.utils.functions import load_file_to_df
# from project.topic_modeling.topmod_weights2df import Topic_Model_Class
from project.journal_summarization.LSTM_summarizer import LSTMSummarizer
from project.journal_summarization.abstract_summarization import AbstractSummarizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


class ProfileGenerator:

    def __init__(self, num_articles):
        self.num_journals = num_articles
        # self.moral_foundations = self._collect_moral_foundations()
        self.topics = self._collect_topics()


    def _collect_topics(self):
        '''
        Collects all 10 topics.

        Returns:  (df) of lists of words comprising each topic.
        '''
        # model_weights_file_path = 'project/topic_modeling/reddit_all_comments_10_topmod'
        # topic_model = Topic_Model_Class(model_weights_file_path)
        # topics_df = topic_model.weights2df()
 
        # indices_of_topics_to_keep = [0, 1, 2, 5, 7, 9, 10, 12, 16, 18]
        # topic_model_rows_df = topics_df.loc[indices_of_topics_to_keep]
        # topic_words_column = topic_model_rows_df.Representation
        topic_model_rows_df = pd.read_csv('project/data/topics.csv')
        return topic_model_rows_df
    

    def _collect_moral_foundations(self, topic):
        '''
        Collects top 2 moral foundations.

        Returns: (list) of moral foundations
        '''
        topics = pd.read_csv('project/data/topics.csv')
        mf1 = topics.loc[topics['Representation	'] == topic,'Dominant_Moral_Foundation'].iloc[0]
        mf2 = topics.loc[topics['Representation	'] == topic, 'Second_Dominant_Moral_Foundation'].iloc[0]
        moral_founds = []
        moral_founds.append(mf1)
        moral_founds.append(mf2)
        return moral_founds


    def _collect_keywords(self, article_data_path):
        '''
        Collects keywords from journal articles to retreive keywords.

        Inputs:
            article_data_path (str): path to cleaned artcile data
        
        Returns: (df column) of list of keyword terms for each article
        '''
        article_data =  pd.read_feather(article_data_path)
        keyword_df = article_data[['doi','key_terms']]
        return keyword_df
    

    def _create_topic_vector(self, model, topic_words):
        '''
        Creates average topic words vector

        Inputs:
            model: gensim Word2Vec model object
            topic_words (list): list of words for one topic

        Returns: single vector
        '''
        vectors = np.array([model[word] for word in topic_words if \
                            word in model])
        concat_topic_vector = np.mean(vectors, axis=0) 
        return concat_topic_vector


    def _create_keyword_vector(self, keywords, model):
        '''
        Creates average key words vector

        Inputs:
            model: gensim Word2Vec model object
            keywords (array): words for one article

        Returns (array): single vector
        '''
        vectors = np.array([model[keyword] for keyword in keywords if \
                            keyword in model])
        concat_vector = np.mean(vectors, axis=0) 
        return concat_vector
    

    def _create_all_keyword_vectors(self, model, keyword_df):
        '''
        Creates df of keyword vectors for all articles.
        Inputs:
            model: gensim Word2Vec model object
            keyword_df (df): all articles and keywords
        
        Returns (df): keywords_vec_df
        '''
        keywords_vec_df = keyword_df
        keywords_vec_df['keyword_vectors'] = keyword_df.loc[keyword_df['key_terms'].notnull(),'key_terms'].apply(
                            lambda x: self._create_keyword_vector(x, model))

        return keywords_vec_df


    def _calc_cosine_similarity(self, topic_vec, keyword_vec):
        '''
        Calcultes cosine similarity for two individual vectors.

        Inputs:
            topic_vec: single vector of avg topic words
            keyword_vec: single vector of avg key words

        Returns (int64): cosine similarity
        '''
        sim = np.dot(topic_vec, keyword_vec) / \
            (np.linalg.norm(topic_vec) * np.linalg.norm(keyword_vec))
        
        int_similarity = np.round(sim * 1000000).astype(int)
        if not isinstance(int_similarity, np.int64):
            int_similarity = None
        return int_similarity


    def _select_articles(self, topic_vector, keywords_vec_df, article_data_path):
        """
        Selects top num_articles articles based on cosine similarity between 
        topic and keywords.

        Inputs:
            topic_vector: single vector of avg topic words
            keywords_vec_df: contains all articles and keywords
            article_data_path (str): path to clean article data
        """
        keywords_vec_df['cosine_sim'] = keywords_vec_df['keyword_vectors'].apply(
                                        partial(self._calc_cosine_similarity, 
                                                topic_vector))
        keywords_vec_df['cosine_sim'] = pd.Series(keywords_vec_df['cosine_sim'])
        top_article_df = keywords_vec_df.nlargest(self.num_journals, 
                                                  'cosine_sim')

        article_data = pd.read_feather(article_data_path)

        article_data_unique = article_data.drop_duplicates(subset=['doi'])
        top_article_df_unique = top_article_df.drop_duplicates(subset=['doi'])
        top_articles = article_data_unique.loc[article_data_unique['doi'].isin(top_article_df_unique['doi'])]

        return top_articles


    # def _make_long_summary(self, document):
    #     summarizer = LSTMSummarizer(dev_environment=False, 
    #                                 load_presaved_model=True)
        
    #     summary = summarizer.summarize(document)
    #     return summary


    def _make_short_summary(self, document):
        '''
        Creates short summaries of article abstracts.

        Input:
            document (str): article abstratcs

        Returns (str): summary of abstract
        '''
        model_path = './project/data/models/small_weights/'
        summarizer = AbstractSummarizer('hi', 
                                        model_path, 
                                        read_from_external = True)
        summary = summarizer.summarize(document)
        return summary


    def generate_printout(self, moral_foundations, top_articles, topic_words, topic_idx):
        """
        Create a printout for each of the top 10 topics discussed pertaining to climate change.

        Inputs:
            moral_foundations (list): top 2 moral foundations
            top_articles (df): top  num_articles articles
            topic_words (list):  list of words creating topic
            topic_idx (int): topic number out of 10

        Output (str): containing text for toolkit PDFs
        """
        moral_foundation1 = moral_foundations[0]
        moral_foundation2 = moral_foundations[1]

        # Create Output
        title = f"TOPIC {topic_idx + 1}\n \n"
        topic_rep = f"The representative words for this topic area are: \n {topic_words} \n \n"
        # top two moral foundations to focus on
        moral_founds = f"The top two moral foundations to focus on when communicating about this topic area are: \n {moral_foundation1} and {moral_foundation2} \n \n"
        # top self.num_journals of journal articles you can read to learn more about the topic: REVISE
        num_sum = f"The top {self.num_journals} journal articles most representative of this topic area with their short and long summaries are: \n \n"
        # make summaries for each article
        output = title + topic_rep + moral_founds + num_sum
 
        for i in range(len(top_articles)):
            if top_articles.loc[i, 'abstract']:
                short_summary_string = self._make_short_summary(top_articles.loc[i, 'abstract'])
                start_idx = next(i for i, c in enumerate(short_summary_string) if c != ' ')
                end_idx = next(i for i, c in enumerate(short_summary_string[::-1]) if c != ' ')
                short_summary = short_summary_string[start_idx:len(short_summary_string)-end_idx]

                # long_summary_string = self._make_long_summary(top_articles)
                short_sum = f"\n {short_summary} \n ."
                # long_sum = f"Long summary: \n {long_summary_string} \n ."
                output += short_sum
                # output += long_sum

        return output


    def _create_pdf_from_string(self, text, filename):
        '''
        Builds and saves PDF from toolkit output string.

        Inputs:
            text (str): text to be printed on PDF
            filename (str): path to and name of PDF to be saved
        '''
        c = canvas.Canvas(filename, pagesize=letter)

        #Format
        width, height = letter
        c.setFont("Helvetica", 12)
        lines = text.split('\n')

        x = 50
        y = height - 50
        max_width = width - 2 * x

        # Write each line of text to the PDF
        for line in lines:
            words = line.split()
            current_line = ''
            for word in words:
                if c.stringWidth(current_line + ' ' + word) < max_width:
                    current_line += ' ' + word
                else:
                    c.drawString(x, y, current_line.strip())
                    y -= 20 
                    current_line = word
            c.drawString(x, y, current_line.strip())
            y -= 20
        c.save()


    def create_toolkits(self):
        '''
        Creates PDF toolkits from multiple project outputs.
        '''
        model = api.load("glove-wiki-gigaword-50")
        article_data_path = 'project/data/proquest_data_cleaned.fea'
        keyword_df = self._collect_keywords(article_data_path)
        keywords_vec_df = self._create_all_keyword_vectors(model, keyword_df)
        topics = self._collect_topics()

        i = 0
        for i, topic in enumerate(topics['Representation	']):
            topic_vector = self._create_topic_vector(model, topic)
            if i <10:
                # topic_vector = np.random.rand(50) # for testing only
                # profile_info = self._generate_profile(model, topic)
                selected_articles = self._select_articles(topic_vector, 
                                                    keywords_vec_df, 
                                                    article_data_path).reset_index()
                moral_foundations = self._collect_moral_foundations(topic)
                single_printout = self.generate_printout(moral_foundations, selected_articles, topic, i)
                self._create_pdf_from_string(single_printout, f'project/data/toolkits/Topic{i+1}_Toolkit.pdf')
                i+=1
            else:
                break
        return


    def __repr__(self):
        s = f"Topics: {self.topics}\n \
        Number of articles: {self.num_journals}"
        
        return s
    

generator = ProfileGenerator(3)
profile = generator.create_toolkits()
print(profile)