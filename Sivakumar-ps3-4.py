'''
Problem Set 3 and 4 for Applied Machine Learning, Spring 2017
Rajeswari Sivakumar
4/27/17

1. Text processing with movie reviews
    a. Remove punctuation
    b. Put all words in lower case
    c. Stem and remove stop words
    d. Create a document term matrix which only uses text frequencies
    e. Train a naive Bayes classifier using DTM with 85/15 training/test split
    f. Calculate and report confusion matrix, accuracy, specificity and sensitivity of classifier.
       Save trained classifier as object, "trainedNBclassifier"
    g. Train a SVM classifier with 85/15 training/test split
    h. Calculate and report confusion matrix, accuracy, specificity and sensitivity of classifier.
       Save trained classifier as object, "trainedSVMclassifier"

2. Use trained classifiers from 1 to classify tweets in pelosi-Sivakumar.csv. Report:
    a. Percent pos/neg
    b. Plot histogram of pos probabilities for all pelosi tweets for each classifier
3. Estimate k=5 topic model for movie reviews data; report top 10 words
4. Estimate k=10 topic model for movie reviews data; report top 10 words
5. Estimate perplexity of k=2 ...10 topic models. Plot perplexity versus each topic.
   Based on perplexity score and observing each topic coherence, print which model should be selected
6. Using topic models above, calculate which reviews are most similar to reviews 1, 2, 3

'''
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import string
import csv
import sys
import matplotlib

def process_text(input_str):
    '''
    Remove punctuation;
    put all words in lower case;
    Stem and remove stop words
    '''
    #remove extra whitespace
    out = ' '.join(input_str.split())
    #lower case
    out = out.lower()
    for ch in string.punctuation:
        out = out.replace(ch, "")

    filtered_words = [word for word in out.split() if word not in stopwords.words('english')]
    return filtered_words

def create_dtm():
    '''
    create text document matrix for each data set with top 10 frequently used words
    '''
    movies = pd.read_csv('movie-pang02.csv')
    word_list = []

    for row in movies['text']:
        rev_words = process_text(row)
        word_list += rev_words
    word_count = np.zeros((2000,len(word_list)), dtype = np.int)
    for i, row in enumerate(movies['text']):
        filtered_words = process_text(row)
        for j, word in enumerate(word_list):
            word_count[i,j] = filtered_words.count(word)
    print(word_count)

if __name__ == '__main__':
    create_dtm()
