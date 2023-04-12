import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

import pyarabic.araby as araby
from nltk.corpus import stopwords
from arabicstopwords.arabicstopwords import stopwords_list
from nltk.stem.snowball import ArabicStemmer
from qalsadi.lemmatizer import Lemmatizer
from flask import Flask, render_template, request

# Load necessary data and models
df = pd.read_csv(r"DeployingData/processed_df.csv")

def normalize_chars(txt):
    # Normalize Arabic characters to simplify text processing
    text = re.sub("[إأآا]", "ا", txt)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub(r'(.)\1+', r'\1', text)
    return text

def clean_txt(txt, stopwordlist, lemmer):
    # normalize chars
    txt = normalize_chars(txt)
    # remove stopwords & punctuation
    txt = ' '.join([token.translate(str.maketrans('','',string.punctuation)) for token in txt.split(' ') if token not in stopwordlist])
    # lemmatizer
    txt_lemmatized = ' '.join([lemmer.lemmatize(token) for token in txt.split(' ')])
    return txt + " " + txt_lemmatized

def show_best_results(df_quran, scores_array, top_n=20):
    results = []
    count=0
    sorted_indices = scores_array.argsort()[::-1]

    for position, idx in enumerate(sorted_indices[:top_n]):
        row = df_quran.iloc[idx]
        data_out = row['clean_txt']
        score = scores_array[idx]
        count +=1
        if score > 0:
            result_dict = {"data_out": data_out}
            results.append(result_dict)
            if  count==3:
                break
    return results

# Extract the clean text from the dataframe
corpus = df['clean_txt']

# Instantiate the vectorizer object
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Vectorize the corpus
corpus_vectorized = vectorizer.fit_transform(corpus)

def run_arabic_search_engine(query):

    stopwordlist = set(list(stopwords_list()) + stopwords.words('arabic'))
    stopwordlist = [normalize_chars(word) for word in stopwordlist]
    st = ArabicStemmer()
    lemmer = Lemmatizer()

    # Preprocess the query
    query = clean_txt(query, stopwordlist, lemmer)

    query_vectorized = vectorizer.transform([query])
    scores = query_vectorized.dot(corpus_vectorized.transpose())
    scores_array = scores.toarray()[0]

    # Return the results as a list of dictionaries
    return show_best_results(df, scores_array)


app = Flask(__name__)

@app.route('/')
def search():
    return render_template('search.html')

@app.route('/results', methods=['POST'])
def results():
    query = request.form['query']
    results = run_arabic_search_engine(query)
    return render_template('results.html', query=query, results=results)
if __name__ == '__main__':
    app.run(debug=True)