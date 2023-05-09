from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.tokenize import word_tokenize

class WordsCount(BaseEstimator, TransformerMixin):
    '''A custom transformer for counting the words in a text

    Methods
    -------
    wordsCount(text: string)
        Returns the number of words in the given text

    fit(X, y)
        Fits the data

    transform(X)
        Applies wordsCount function over X
    '''
    def wordsCount(self, text):
        '''Counts and returns the number of words in a text

        Args:
            text (string): The text to be processed

        Returns:
            The number of words in the given text
        '''
        return len(word_tokenize(text))
    
    def fit(self, X, y=None):
        '''Fits the data
        '''
        return self
    
    def transform(self, X):
        '''Transforms the data - applies wordsCount on every item of the list

        Args:
            X: a list of texts

        Returns:
            A list of numbers representing the number of words in each text
        '''
        X_tagged = pd.Series(X).apply(self.wordsCount)
        return pd.DataFrame(X_tagged)
    
