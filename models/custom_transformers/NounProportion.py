import nltk
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from utils import tokenize

class NounProportion(BaseEstimator, TransformerMixin):
    def nounProportion(self, text):
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
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.nounProportion)
        return pd.DataFrame(X_tagged)
    
