import nltk
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from utils import tokenize

class NounProportion(BaseEstimator, TransformerMixin):
    '''A custom transformer for calculating the proportion of nouns in a text

    Methods
    -------
    nounProportion(text: string)
        Returns the calculated proportion of nouns in a text.
        Uses 'tokenize' function for tokenizint a text.

    fit(X, y)
        Fits the data

    transform(X)
        Applies nounProportion function over X
    '''
    def nounProportion(self, text):
        '''Calculates and returns the proportion of nouns in a text

        Args:
            text (string): The text to be processed

        Returns:
            The proportion of nouns for the given text
        '''
        words_list = tokenize(text)

        if len(words_list) == 0:
            return 0

        pos_tags = nltk.pos_tag(words_list)
        counter = 0
        for word, tag in pos_tags:
            print(word, tag)
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                counter += 1
        
        return counter / len(words_list)

    
    def fit(self, X, y=None):
        '''Fits the data
        '''
        return self
    
    def transform(self, X):
        '''Transforms the data - applies nounProportion on every item of the list

        Args:
            X: a list of texts

        Returns:
            A list of numbers representing the proportion of nouns in each text
        '''
        X_tagged = pd.Series(X).apply(self.nounProportion)
        return pd.DataFrame(X_tagged)
    
