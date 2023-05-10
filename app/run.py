from transformers.WordsCount import WordsCount
from transformers.NounProportion import NounProportion
from transformers.CapitalWordsCount import CapitalWordsCount
from utils.tokenize import tokenize
import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesCategories', con=engine.connect())

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.drop(
        columns=['id', 'message', 'original', 'genre']).columns
    category_counts = df[category_names].sum().sort_values()

    direct = df[df['genre'] == 'direct'][category_names].sum(
        axis=1).value_counts().sort_index()
    news = df[df['genre'] == 'news'][category_names].sum(
        axis=1).value_counts().sort_index()
    social = df[df['genre'] == 'social'][category_names].sum(
        axis=1).value_counts().sort_index()

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {

            'data': [
                Bar(
                    x=category_counts.index,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Number of messages by category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                Bar(
                    name="direct",
                    x=direct.index,
                    y=direct
                ),
                Bar(
                    name="news",
                    x=news.index,
                    y=news
                ),
                Bar(
                    name="social",
                    x=social.index,
                    y=social
                )
            ],

            'layout': {
                'title': 'Number of messages by the number of categories assigned, grouped by genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of categories assigned"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    print(classification_labels)
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
