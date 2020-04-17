# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:02:51 2020

@author: Zach
"""

import numpy
import pickle as pkl

from collections import OrderedDict
from nltk.corpus import stopwords

import glob
import os
import re
import string


dataset_path='/ES_Final/Data/'

def extract_words(sentences):
    result = []
    stop = stopwords.words('english')
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


def grab_data(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = extract_words(sentences)

    return sentences

def main():
    path = dataset_path

    train_x_pos = grab_data(path+'train/pos')
    train_x_neg = grab_data(path+'train/neg')
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos')
    test_x_neg = grab_data(path+'test/neg')
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('train.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    f.close()
    f = open('test.pkl', 'wb')
    pkl.dump((test_x, test_y), f, -1)
    f.close()


if __name__ == '__main__':
    main()
