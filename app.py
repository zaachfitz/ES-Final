# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:29:05 2020

@author: Zach
"""


from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib
#from nltk.corpus import stopwords
import re

app = Flask(__name__)



@app.route('/', methods=['GET'])
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Load All Reviews in train and test datasets
    f = open('train.pkl', 'rb')
    reviews = pickle.load(f)
    f.close()

    f = open('test.pkl', 'rb')
    test = pickle.load(f)
    f.close()
    def extract_words(sentences):
        result = []
        #stop = stopwords.words('english')
        unwanted_char = '?.,!:;"$%^&*()#@+/0123456789<>=\\[]_~{}|`'
        trans = str.maketrans(unwanted_char, ' '*len(unwanted_char))

        for s in sentences:
            s = re.sub(r"([?.!;,:()\"])", r" \1 ", s) #take first argument it matches and puts a space after it. inside [] is what we want to match
    
            s = re.sub(r'[" "]+', " ", s) #replace spaces if you want to get rid of tabs it's \t. this replaces with single space.
    
            s = re.sub(r"[^a-zA-Z?.!;:,()\"]+", " ", s) #just removes anything that we do not want. exluding what is in []
    
            s = s.rstrip().strip()  #just remove anything at the start that we dont want.

            words = []
            for word in s.split():
                word = word.lstrip('-\'\"').rstrip('-\'\"')
                if len(word)>2 :
                    words.append(word.lower())
                    s = ' '.join(words)
                    result.append(s.strip())
        return result
    # Generate counts from text using a vectorizer.  
    # This performs step of computing word counts.
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, 
                            sublinear_tf=True, use_idf=True)
    train_features = vectorizer.fit_transform(reviews[0])
    test_features = vectorizer.transform(test[0])
    sent_classifier = open('sent_classifier.pkl', 'rb')
    clf = joblib.load(sent_classifier)


    
    if request.method == 'POST':
        sentences = []
        sentence = request.form['sentence']
        sentences.append(sentence)
        input_features = vectorizer.transform(extract_words(sentences))
        prediction_new = clf.predict(input_features)
        #if prediction[0] == 1 :
            #print("---- \033[92mpositive\033[0m\n")
        #else:
            #print("---- \033[91mneagtive\033[0m\n")
    return render_template('result.html', prediction = prediction_new)


if __name__ == '__main__':
    app.run()