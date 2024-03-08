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
from project.utils.functions import load_file_to_df
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
        Collects all 10 topics and returns df of lists of words comprising 
        each topic.
        '''
        # model_weights_file_path = 'project/topic_modeling/reddit_all_comments_10_topmod'
        # topic_model = Topic_Model_Class(model_weights_file_path)
        # topics_df = topic_model.weights2df()
 
        # indices_of_topics_to_keep = [0, 1, 2, 5, 7, 9, 10, 12, 16, 18]
        # topic_model_rows_df = topics_df.loc[indices_of_topics_to_keep]
        # topic_words_column = topic_model_rows_df.Representation
        topic_model_rows_df = pd.read_csv('project/data/topics.csv')
        return topic_model_rows_df
        
        return pd.DataFrame({'name': ['topic1', 
                                      'topic2'],
                             'Representation': [['climate', 'ocean', 'heat'], 
                                                ['meat', 'animal', 'vegeterian']]})

    def _collect_moral_foundations(self, topic):
        '''
        Returns list of top 2 moral foundations
        '''
        # TODO: call moral foundtions and select top 2
        topics = pd.read_csv('project/data/topics.csv')
        mf1 = topics.loc[topics['Representation	'] == topic,'Dominant_Moral_Foundation'].iloc[0]
        mf2 = topics.loc[topics['Representation	'] == topic, 'Second_Dominant_Moral_Foundation'].iloc[0]
        moral_founds = []
        moral_founds.append(mf1)
        moral_founds.append(mf2)
        return moral_founds

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
        # words = [word for word in topic_words if word != ' ']
        print("WORDS:")
        # print(words)
        vectors = np.array([model[word] for word in topic_words if \
                            word in model])

        concat_topic_vector = np.mean(vectors, axis=0) 
        # if (len(vectors) > 0) & \
        #     (vectors is not None) else np.zeros(model.vector_size)
      
        return concat_topic_vector

    def _create_keyword_vector(self, keywords, model):
        # Vectorize words in the word list
        # keywords = [keyword for keyword in keywords if keyword.strip()]
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
        keywords_vec_df['cosine_sim'] = keywords_vec_df['keyword_vectors'].apply(
                                        partial(self._calc_cosine_similarity, 
                                                topic_vector))
        # non_numeric_values = [type(x) for x in keywords_vec_df['cosine_sim'] if not isinstance(x, np.int64)]
       
        print('TOPIC VECTOR')
        print(topic_vector)
        print()

        print(keywords_vec_df[:2])
        
        keywords_vec_df['cosine_sim'] = pd.Series(keywords_vec_df['cosine_sim'])

        # top_article_df = keywords_vec_df.dropna(subset=['cosine_sim']
        #                                 ).sort_values(by = 'cosine_sim',
        #                                             ascending=False,
        #                                             inplace = True
        #                                 ).head(self.num_journals)
        
        top_article_df = keywords_vec_df.nlargest(self.num_journals, 
                                                  'cosine_sim')
        print(top_article_df[:1])
        print(len(top_article_df))
        
        article_data = pd.read_feather(article_data_path)

        article_data_unique = article_data.drop_duplicates(subset=['doi'])
        print(f"NUMBER OF UNIQUE ARTICLES: {len(article_data_unique)}")
        top_article_df_unique = top_article_df.drop_duplicates(subset=['doi'])
        top_articles = article_data_unique.loc[article_data_unique['doi'].isin(top_article_df_unique['doi'])]
        # top_article_df_subset = top_article_df_unique.loc[top_article_df_unique['doi'].isin(article_data_unique['doi'])]

        # top_articles = article_data.loc[article_data['doi'].drop_duplicates().reset_index().isin(
        #                                         top_article_df['doi'].unique())]
        # top_articles = top_articles.loc[top_articles['doi']]
        print(top_articles.columns)
        print(f"Number of articles:{len(top_articles)}")
        print(top_articles[:1])
        print(f"Number should be: {self.num_journals}")
        return top_articles

    # def _make_long_summary(self, document):
    #     summarizer = LSTMSummarizer(dev_environment=False, 
    #                                 load_presaved_model=True)
        
    #     summary = summarizer.summarize(document)
    #     return summary


    def _make_short_summary(self, document):
        model_path = './project/data/models/small_weights/'
        # print(document)
        # print(os.listdir(model_path))
        summarizer = AbstractSummarizer('hi', 
                                        model_path, 
                                        read_from_external = True)
        summary = summarizer.summarize(document)
        return summary


    def generate_printout(self, moral_foundations, top_articles, topic_words, topic_idx):
        """
        Create a printout for each of the top 10 topics discussed pertaining to climate change.
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
        output = topic_rep + moral_founds + num_sum
        print(type(top_articles))
        print(len(top_articles))
        for i in range(len(top_articles)):
            print('ARTICLE:', i)
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
        for topic in topics.iterrows()['Representation	']:
            profile_info = self._generate_profile(model, topic)
            moral_foundations, top_articles = profile_info
            self.generate_printout(moral_foundations, top_articles, topic)


    def _create_pdf_from_string(self, text, filename):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        # Set the font and size
        c.setFont("Helvetica", 12)

        # Split the text into lines
        lines = text.split('\n')

        # Set the starting position for the text
        x = 50
        y = height - 50

        # Set the maximum width for the text
        max_width = width - 2 * x

        # Write each line of text to the PDF
        for line in lines:
            # Split the line into words
            words = line.split()
            current_line = ''
            for word in words:
                # Check if adding the word exceeds the maximum width
                if c.stringWidth(current_line + ' ' + word) < max_width:
                    # Add the word to the current line
                    current_line += ' ' + word
                else:
                    # Draw the current line and start a new line with the word
                    c.drawString(x, y, current_line.strip())
                    y -= 20  # Move down by 20 units for the next line
                    current_line = word
            c.drawString(x, y, current_line.strip())
            y -= 20  # Move down by 20 units for the next line
        # Save the PDF
        c.save()



    def create_toolkits(self): #this is generate_profile()
        model = api.load("glove-wiki-gigaword-50")
        print("Word2Vec model loaded")
        article_data_path = 'project/data/proquest_data_cleaned.fea'
        keyword_df = self._collect_keywords(article_data_path)
        keywords_vec_df = self._create_all_keyword_vectors(model, keyword_df)

        topics = self._collect_topics()

        print(topics['Topic'])

        i = 0
        for i, topic in enumerate(topics['Representation	']):
            print(topic)
            topic_vector = self._create_topic_vector(model, topic)
            print(topic_vector)
            print(len(topic_vector))
            print()
            if i <10:
                print(f"i ===== {i}")
                # topic_vector = np.random.rand(50) # for testing only
                # profile_info = self._generate_profile(model, topic)
                selected_articles = self._select_articles(topic_vector, 
                                                    keywords_vec_df, 
                                                    article_data_path).reset_index()
                print(selected_articles)
                moral_foundations = self._collect_moral_foundations(topic)
                single_printout = self.generate_printout(moral_foundations, selected_articles, topic, i)
                print(single_printout)
                self._create_pdf_from_string(single_printout, f'project/data/toolkits/Topic{i+1}_Toolkit.pdf')
                i+=1
            else:
                break
            


        

        
        # return selected_articles
        return

        # topics = self._collect_topics()
        # for topic in topics:
        #     print(self.generate_profile(model, topic))



    # def __repr__(self):
    #     s = f"Topics: {self.topics}\n \
    #     Number of articles: {self.num_journals}"
        
    #     return s
    

generator = ProfileGenerator(3)
profile = generator.create_toolkits()
print(profile)