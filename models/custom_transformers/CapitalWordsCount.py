import nltk
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from utils import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class CapitalWordsCount(BaseEstimator, TransformerMixin):
    def capitalWordsCount(self, text):
        words_list = word_tokenize(text)
        stopwords_list = stopwords.words('english')

        words_list = [word for word in words_list if word not in stopwords_list]
        words_list = [word for word in words_list if word[0].isupper()]

        print(words_list)
        return len(words_list)

    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.capitalWordsCount)
        return pd.DataFrame(X_tagged)
    
