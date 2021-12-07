import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pkg_resources
from symspellpy import SymSpell, Verbosity
import nltk
import json
import os
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoders.default(self, obj)

def modalFunction(text):
    yelp = pd.read_csv('yelp.csv')
    yelp.head()
    yelp['text length'] = yelp['text'].apply(len)
    sns.set_style('white')
    g = sns.FacetGrid(yelp,col='stars')
    g.map(plt.hist,'text length')
    sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
    sns.countplot(x='stars',data=yelp,palette='rainbow')
    stars = yelp.groupby('stars').mean()
    stars.corr()
    yelp_class = yelp
    X = yelp_class['text']
    y = yelp_class['stars']
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
    nb = MultinomialNB()
    nb.fit(X_train,y_train)
    predictions = nb.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print('\n')
    print(classification_report(y_test,predictions))
    inp=[text]
    b=cv.transform(inp)
    out=nb.predict(b)
    return out

# app
app = Flask(__name__)
# routes
@app.route('/gg/<text>', methods=['GET'])

def predict(text):
    # output = {'results': 123}
    # text="Hello i am Doctor with chills stomachach kidney failure paracetamol headache dolo easy";
    output=modalFunction()
    print(output)
    return jsonify(json.dumps({"results":output}, cls=NumpyArrayEncoder))

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/rating', methods=['POST'])

def rating():
    # data = request.form.get("name")
    data = request.get_json() 
    # data2 = request.form.get("gg")
    # data3 = request.form.get("num")
    # print(data,data2,data3)
    print(data['text'])
    output=modalFunction(data['text'])
    print(output)
    return jsonify(json.dumps({"result":output}, cls=NumpyArrayEncoder))
    # return jsonify("Hello")

if __name__ == '__main__':
    app.run(port = 5000, debug=True)