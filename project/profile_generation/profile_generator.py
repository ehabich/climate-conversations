#!/usr/bin/env python3
'''
Generates summaries of articles based on categories of interest.

Authors: Kate Habich, Jen Yeaton
'''
import project.data
import 

class ProfileGenerator:

    def __init__(self, num_journals):
        self.num_journals = num_journals
        self.moral_foundations = self._collect_moral_foundations()
        self.topics = self._collect_topics()

    def _collect_moral_foundations(self):
        return 'moral foundation 1'

    def _collect_topics(self):
        return 'topic 1'

    def _collect_keywords():

        return keywords

    def _create_category_vector():
        pass

    def _create_keyword_vector():
        pass

    def _calc_cosine_similarity():
        pass

    def _select_journals():
        # pass in self.num_journals
        pass

    def _make_long_summary():
        summarizer = LSTMSummarizer(dev_environment=False, load_presaved_model=True)

    def _make_short_summary():
        
        pass


    def generate_profile():
        pass


    def __repr__(self):
        s = f"Moral Foundations: {self.moral_foundations}\n \
        Topics: {self.topics}\n \
        Number of articles: {self.num_journals}"
        
        return s
    

profile = ProfileGenerator(3)
print(profile)