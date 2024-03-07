#!/usr/bin/env python3
'''
Generates summaries of articles based on categories of interest.

Authors: Kate Habich, Jen Yeaton
'''
import project.data
import numpy as np
import pandas as pd
from functools import partial
from project.utils.functions import load_file_to_df
# from project.topic_modeling.topmod_weights2df import Topic_Model_Class
from project.journal_summarization.LSTM_summarizer import LSTMSummarizer
from project.journal_summarization.abstract_summarization import AbstractSummarizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api


class ProfileGenerator:

    def __init__(self, num_articles):
        self.num_journals = num_articles
        self.moral_foundations = self._collect_moral_foundations()
        self.topics = self._collect_topics()


    def _collect_topics(self):
    #     '''
    #     Collects all 10 topics and returns df of lists of words comprising 
    #     each topic.
    #     '''
    #     model_weights_file_path = 'project/topic_modeling/reddit_all_comments_10_topmod'
    #     topic_model = Topic_Model_Class(model_weights_file_path)
    #     topics_df = topic_model.weights2df()
    #     topic_words_column = topics_df.Representation
    #     return topic_words_column
        
        return pd.DataFrame({'name': ['topic1', 
                                      'topic2'],
                             'Representation': [['climate', 'ocean', 'heat'], 
                                                ['meat', 'animal', 'vegeterian']]})

    def _collect_moral_foundations(self):
        '''
        Returns list of top 2 moral foundations
        '''
        # TODO: call moral foundtions and select top 2
        return ['moral foundation 1', 'moral foundation2']

    def _collect_keywords(self, article_data_path):
        '''
        Collects keywords from journal articles and returns df column 
        containing list of keyword terms for each article.
        '''
        article_data =  pd.read_feather(article_data_path)
        keyword_df = article_data[['doi','key_terms']]
        return keyword_df

    def _create_topic_vector(self, model, topic_words):
        '''
        Assumed to be passing in list of words making up a sincle topic.
        '''
        words = [word for word in words if word.strip()]
        vectors = np.array([model[words] for word in words if \
                            word in model])

        concat_topic_vector = np.mean(vectors, axis=0) 
        # if (len(vectors) > 0) & \
        #     (vectors is not None) else np.zeros(model.vector_size)
      
        return concat_topic_vector

    def _create_keyword_vector(self, keywords, model):
        # Vectorize words in the word list
        keywords = [keyword for keyword in keywords if keyword.strip()]
        vectors = np.array([model[keyword] for keyword in keywords if \
                            keyword in model])

        concat_vector = np.mean(vectors, axis=0) 
        # if (len(vectors) > 0) & (vectors is not None) else np.zeros(model.vector_size)
      
        return concat_vector
    
    def _create_all_keyword_vectors(self, model, keyword_df):
        keywords_vec_df = keyword_df
        keywords_vec_df['keyword_vectors'] = keyword_df.loc[keyword_df['key_terms'].notnull(),'key_terms'].apply(
                            lambda x: self._create_keyword_vector(x, model))

        return keywords_vec_df

    def _calc_cosine_similarity(self, topic_vec, keyword_vec):
        '''
        Calcultes cosine similarity for two individual vectors.
        '''
        sim = np.dot(topic_vec, keyword_vec) / \
            (np.linalg.norm(topic_vec) * np.linalg.norm(keyword_vec))
        
        int_similarity = np.round(sim * 1000000).astype(int)
        if not isinstance(int_similarity, np.int64):
            # print("found non-int")
            int_similarity = None
        return int_similarity

    def _select_articles(self, topic_vector, keywords_vec_df, article_data_path):
        """
        Selects top num_articles articles based on cosine similarity between 
        topic and keywords.
        """
        print("cosine")
        # print(keywords_vec_df)
        # print(topic_vector)
        print()
        # print(keywords_vec_df.columns)
        print(keywords_vec_df.loc[0,'keyword_vectors'])
        
        print('done')
        keywords_vec_df['cosine_sim'] = keywords_vec_df['keyword_vectors'].apply(
                                        partial(self._calc_cosine_similarity, 
                                                topic_vector))
        non_numeric_values = [type(x) for x in keywords_vec_df['cosine_sim'] if not isinstance(x, np.int64)]
        # print("Non-numeric values in 'cosine_sim':", non_numeric_values)
        print(non_numeric_values)
        print(len(non_numeric_values))
        
        # keywords_vec_df['cosine_sim'] = keywords_vec_df['keyword_vectors'].apply(
        #                                 partial(isinstance, 
        #                                         int))
        # print(keywords_vec_df.loc[not keywords_vec_df['cosine_sim'], :])
        # keywords_vec_df = keywords_vec_df.drop(9675)
        # keywords_vec_df = keywords_vec_df[keywords_vec_df['cosine_sim'].apply(lambda x: not isinstance(x, np.ndarray))]

        
        # print(keywords_vec_df.loc[0:3,'cosine_sim'])
        # print(keywords_vec_df.columns)
        # print("weird type:", type(keywords_vec_df.loc[9675, 'cosine_sim']))
        # print(keywords_vec_df.loc[9675, 'cosine_sim'])
        # article_data_w_cosine_sim = pd.concat([keywords_vec_df, cosine_sim_calcs], 
        #                                       axis=1)
        # top_articles = article_data_w_cosine_sim.sort_values(by="cosine_sim_calc", 
        #                                                      ascending=False).head(self.num_journals)
        # keywords_vec_df['cosine_sim'] = pd.to_numeric(keywords_vec_df['cosine_sim'])
        print(type(keywords_vec_df))
        
        keywords_vec_df['cosine_sim'] = pd.Series(keywords_vec_df['cosine_sim'])

        # top_article_df = keywords_vec_df.dropna(subset=['cosine_sim']
        #                                 ).sort_values(by = 'cosine_sim',
        #                                             ascending=False,
        #                                             inplace = True
        #                                 ).head(self.num_journals)
        
        top_article_df = keywords_vec_df.nlargest(self.num_journals, 
                                                  'cosine_sim')
        
        article_data = pd.read_feather(article_data_path)
        top_articles = article_data.loc[article_data['doi'].isin(
                                                top_article_df['doi'].unique())]
    
        return top_articles

    def _make_long_summary(self, document):
        summarizer = LSTMSummarizer(dev_environment=False, 
                                    load_presaved_model=True)
        
        summary = summarizer.summarize(document)
        return summary


    def _make_short_summary(self, document):
        model_save_path = '/project/models/small_weights/'
        summarizer = AbstractSummarizer(None,
                                model_save_path,
                                read_from_external = True)

        summary = summarizer.summarize(document)
        return summary


    def generate_printout(self, moral_foundations, top_articles, topic_words):
        """
        Create a printout for each of the top 10 topics discussed pertaining to climate change.
        """
        moral_foundation1 = moral_foundations[0]
        moral_foundation2 = moral_foundations[1]

        # Create Output
        topic_rep = f"The representative words for this topic area are {topic_words} \n."
        # top two moral foundations to focus on
        moral_founds = f"The top two moral foundations to focus on when communicating about this topic area are {moral_foundation1} and {moral_foundation2} \n."
        # top self.num_journals of journal articles you can read to learn more about the topic: REVISE
        num_sum = f"The top {self.num_journals} journal articles most representative of this topic area with their short and long summaries are: \n"
        # make summaries for each article
        output = topic_rep + moral_founds + num_sum
        for article in top_articles:
            short_summary_string = self._make_short_summary(top_articles['abstract'])
            long_summary_string = self._make_long_summary(top_articles)
            # short summary of each of the above articles: REVISE
            short_sum = f"Short summary: \n {short_summary_string} \n ."
            # long summary of each of the above articles: REVISE
            long_sum = f"Long summary: \n {long_summary_string} \n ."

            output += short_sum
            output += long_sum

        return output


    def _generate_profile(self, model, topic):
        keyword_df = self._collect_keywords('project/data/proquest_data_cleaned.fea')
        keywords_vec_df = self._create_all_keyword_vectors(model, keyword_df)

        print("journal vecs created")
        topic_vec = self._create_topic_vector(model, topic)

        top_articles = self._select_articles(self, topic_vec, keywords_vec_df)
        moral_foundations = self. _collect_moral_foundations()

        return moral_foundations, top_articles
        return keywords_vec_df
    

    def generate_all_profiles(self):
        """
        Generate profiles by collecting dataframe of articles, word embeddings 
        model, vector representations of key words, and top topics.
        """
        model = api.load("glove-wiki-gigaword-50")
        print("Word2Vec model loaded")
        article_data_path = 'project/data/proquest_data_cleaned.fea'
        keyword_df = self._collect_keywords(article_data_path)
        keywords_vec_df = self._create_all_keyword_vectors(model, keyword_df)

        # TODO: topics should be entire topics dataframe  (nrows = 10, ncols = ~3)
        topics = self._collect_topics()
        for topic in topics.iterrows()['Representation']:
            profile_info = self._generate_profile(model, topic)
            moral_foundations, top_articles = profile_info
            self.generate_printout(moral_foundations, top_articles, topic)



    def test(self): #this is generate_profile()
        model = api.load("glove-wiki-gigaword-50")
        print("Word2Vec model loaded")
        article_data_path = 'project/data/proquest_data_cleaned.fea'
        keyword_df = self._collect_keywords(article_data_path)
        keywords_vec_df = self._create_all_keyword_vectors(model, keyword_df)

        print(keywords_vec_df.loc[0,'keyword_vectors'])

        topic_vector = np.random.rand(50) # for testing only
        selected_articles = self._select_articles(topic_vector, 
                                                  keywords_vec_df, 
                                                  article_data_path)
        
    
        return selected_articles
        # return

        # topics = self._collect_topics()
        # for topic in topics:
        #     print(self.generate_profile(model, topic))





    # def __repr__(self):
    #     s = f"Moral Foundations: {self.moral_foundations}\n \
    #     Topics: {self.topics}\n \
    #     Number of articles: {self.num_journals}"
        
    #     return s
    

generator = ProfileGenerator(3)
profile = generator.test()
print(profile)