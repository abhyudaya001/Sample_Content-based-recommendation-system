from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.tokenize import word_tokenize   # module for tokenizing strings
from nltk.stem import PorterStemmer        # module for stemming
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import re                                  # library for regular expression operations
import string                              # for string operations

from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

data_df = pd.read_csv('train.csv')
data_df.head()

data_df_text = data_df[['Id', 'Title']]
data_df_text.head()


def remove_punctuation(text):
    return "".join(["" if ch in string.punctuation else ch.lower() for ch in text])


stopwords_english = set(stopwords.words('english'))


def clean_words(headline):
    return [
        word for word in headline
        if word not in stopwords_english
    ]


stemmer = PorterStemmer()


def words_stems(headline):
    return [
        stemmer.stem(word) for word in headline
    ]


def tokenize_text(text):
    return word_tokenize(text)


def remove_numbers(text):
    return re.sub("[^a-zA-Z]", " ", text)


data_df_text['Title'] = data_df_text['Title'].apply(remove_punctuation).apply(
    remove_numbers).apply(tokenize_text).apply(clean_words)
data_df_text.head()

tagged_data = [TaggedDocument(row['Title'], [i])
               for i, row in data_df_text.iterrows()]

# model = Doc2Vec(tagged_data, vector_size=20, window=2,
#                 min_count=1, workers=4, epochs=100)

# model.save("st_doc2vec.model")

model = Doc2Vec.load("st_doc2vec.model")


def get_embedding(sentence):
    func_embeddings, func_item_name = [], []
    for word in sentence:

        try:
            vec = model.wv[word]
            func_embeddings.append(vec)
            func_item_name.append(sentence)
        except:
            pass
    return func_embeddings


texts = data_df_text["Title"]
embed_list = []
for text in texts:
    embed_list.append(get_embedding(text))

data_df_text["embeddings"] = embed_list

data_df_text.head()

ques = "overlay an image in CSS"


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict():
    input_string = request.form['input_string']
    predicted_answer = predict_similar_string(input_string)

    return render_template('result.html', answers=predicted_answer)


def predict_similar_string(ques):
    print(ques)
    score_list = []
    for i in range(len(data_df_text)):
        func_embeddings = data_df_text.iloc[i, 2]
        func_embeddings2, func_item_name2 = [], []
        for word in ques.split():

            try:
                vec = model.wv[word]
                func_embeddings2.append(vec)

            except:
                pass
        final_vec2 = [0]*model.wv.vector_size
        for v in func_embeddings2:
            final_vec2 += v

        try:
            score = cosine_similarity(func_embeddings, func_embeddings2)
            score = np.mean(score)
            score_list.append([score, data_df.iloc[i, 1]])
        except:
            pass
    # print(score_list)
    score_list.sort(reverse=True)
    res = []
    try:
        res = [item[1] for item in score_list][:5]
        print(res)
        return res
    except:
        return "nan"


# find_similar_questions("overlay an image in CSS")

if __name__ == "__main__":
    app.run(debug=True)
