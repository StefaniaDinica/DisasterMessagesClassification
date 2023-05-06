import nltk
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from utils import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class CapitalWordsCount(BaseEstimator, TransformerMixin):
    '''A custom transformer for counting the words beginning with a capital letter

    Methods
    -------
    capitalWordsCount(text: string)
        Returns the number of words beginning with a capital letter

    fit(X, y)
        Fits the data

    transform(X)
        Applies capitalWordsCount function over X
    '''
    def capitalWordsCount(self, text):
        '''Returns the number of words beginning with a capital letter

        Args:
            text (string): The text to be processed

        Returns:
            The number of words beginning with a capital letter in the given text
        '''
        words_list = word_tokenize(text)
        stopwords_list = stopwords.words('english')

        words_list = [word for word in words_list if word not in stopwords_list]
        words_list = [word for word in words_list if word[0].isupper()]

        return len(words_list)

    
    def fit(self, X, y=None):
        '''Fits the data
        '''
        return self
    
    def transform(self, X):
        '''Transforms the data - applies capitalWordsCount on every item of the list

        Args:
            X: a list of texts

        Returns:
            A list of numbers representing the number of words beginning with a capital letter
        '''
        X_new = pd.Series(X).apply(self.capitalWordsCount)

        return pd.DataFrame(X_new)
    
CapitalWordsCount.transform