import nltk
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.tokenize import word_tokenize

class WordsCount(BaseEstimator, TransformerMixin):
    def wordsCount(self, text):
        return len(word_tokenize(text))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.wordsCount)
        return pd.DataFrame(X_tagged)
    
