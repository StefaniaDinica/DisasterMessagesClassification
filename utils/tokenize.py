import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)

    words_list = word_tokenize(text)
    stopwords_list = stopwords.words('english')

    words_list = [word for word in words_list if word not in stopwords_list]

    words_list = [lemmatizer.lemmatize(word) for word in words_list]
    words_list = [lemmatizer.lemmatize(word, pos='v') for word in words_list]

    words_list = [stemmer.stem(word) for word in words_list]

    return words_list