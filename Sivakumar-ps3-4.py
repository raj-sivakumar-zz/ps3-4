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
from nltk.corpus import stopwords as sw
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import *

import re
import string
import csv
import sys
import matplotlib
import sklearn as skl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from collections import *
import random as rnd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation

def process_text(input_str):
    '''
    Remove punctuation;
    put all words in lower case;
    Stem and remove stop words
    '''
    output_words = []
    stopwords  = set(sw.words('english'))
    punct = set(string.punctuation)
    url = re.compile('https?://(www.)?([a-z0-9]+)\.[^\s]*')
    tokenizer = TweetTokenizer()
    hashtag = re.compile('#[A-Z]+')
    username = re.compile('@\w+')
    lem = WordNetLemmatizer()

    input_str = input_str.lower()
    input_str = url.sub('',input_str)
    input_str = hashtag.sub('', input_str)
    input_str = username.sub('', input_str)

    token_iter = iter(pos_tag(tokenizer.tokenize(input_str)))
    for token, tag in token_iter:

        # ignore stopwords
        if token in stopwords:
            continue

        # ignore punctuation
        if all(char in punct for char in token):
            continue

        # stem and lemmatize
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        lemma = lem.lemmatize(token, tag)
        output_words.append(lemma)

    return output_words



def dtm(file,n=None):
    '''
    create text document matrix for each data set with top 100 words
    '''
    lines = pd.read_csv(file)
    word_count = skl.feature_extraction.text.CountVectorizer(analyzer=process_text, max_features=n)
    dtm = word_count.fit_transform(lines['text']).toarray()
    features = word_count.get_feature_names()
    output = pd.DataFrame(dtm,columns=features)
    return output

def get_test_train(data, y, train=0.8):
    X = data

    return train_test_split(X,y,test_size=train)
 # return dataframe train

def train_models(file):
    y = pd.read_csv(file)['class']
    X = dtm(file,100)
    X_train,X_test,y_train,y_test = get_test_train(X, y)
    #Naive bayes classifier
    clf_nb =  MultinomialNB().fit(X_train, y_train)
    clf_svm = svm.SVC(probability=True).fit(X_train,y_train)
    Y_nb = clf_nb.predict(X_test)
    Y_svm = clf_svm.predict(X_test)
    print('Naive Bayes:')
    print('Confusion Matrix:\n{}'.format(skl.metrics.confusion_matrix(y_test, Y_nb)))
    tn_nb, fp_nb, fn_nb, tp_nb = skl.metrics.confusion_matrix(y_test, Y_nb).ravel()
    print('Accuracy:{}'.format(skl.metrics.accuracy_score(y_test, Y_nb)))
    print('Specificity:{}'.format(tn_nb/(tn_nb+fp_nb)))
    print('Sensitivity:{}'.format(tp_nb/(tp_nb+fn_nb)))
    print('SVM:')
    print('Confusion Matrix:\n{}'.format(skl.metrics.confusion_matrix(y_test, Y_svm)))
    tn_svm, fp_svm, fn_svm, tp_svm = skl.metrics.confusion_matrix(y_test, Y_svm).ravel()
    print('Accuracy:{}'.format(skl.metrics.accuracy_score(y_test, Y_svm)))
    print('Specificity:{}'.format(tn_svm/(tn_svm+fp_svm)))
    print('Sensitivity:{}'.format(tp_svm/(tp_svm+fn_svm)))


    print('\nPelosi Tweet Stats:')
    X_pelosi = dtm('pelosi-Sivakumar.csv')

    pelosi_nb = list(clf_nb.predict(X_pelosi))
    num_pos_nb = pelosi_nb.count('Pos')
    num_neg_nb = pelosi_nb.count('Neg')
    pos_nb = num_pos_nb/(num_pos_nb+num_neg_nb) *100
    neg_nb = num_neg_nb/(num_pos_nb+num_neg_nb) *100
    print('Naive Bayes Classifier:')
    print('Percent Positive:{}%'.format(pos_nb))
    print('Percent Negative:{}%'.format(neg_nb))
    p_pelosi_nb= clf_nb.predict_proba(X_pelosi)
    plt.figure(1)
    plt.hist(p_pelosi_nb[:,0], 50, normed=0.5, facecolor='green', alpha=0.75)
    plt.xlabel('Probability of Tweet being Pos(NB)')
    plt.ylabel('Frequencies')
    plt.title('H')
    plt.axis([0, 1, 0, 30])
    plt.show()


    pelosi_svm = list(clf_svm.predict(X_pelosi))
    num_pos_svm = pelosi_svm.count('Pos')
    num_neg_svm = pelosi_svm.count('Neg')
    pos_svm = num_pos_svm/(num_pos_svm+num_neg_svm) *100
    neg_svm = num_neg_svm/(num_pos_svm+num_neg_svm) *100
    print('SVM Classifier:')
    print('Percent Positive:{}%'.format(pos_svm))
    print('Percent Negative:{}%'.format(neg_svm))
    p_pelosi_svm = clf_svm.predict_proba(X_pelosi)
    plt.figure(2)
    n, bins, patches = plt.hist(p_pelosi_svm[:,0], 50, normed=0.5, facecolor='blue', alpha=0.75)
    plt.xlabel('Probability of Tweet being Pos(SVM)')
    plt.ylabel('Frequencies')
    plt.title('H')
    plt.axis([0, 1, 0, 30])
    plt.show()

def train_topic_models(file):


    corpus = dtm(file,10000)
    features = corpus.columns.values
    y = pd.read_csv(file)['class']
    lda_5 = LatentDirichletAllocation(n_topics=5, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(corpus)

    lda_10 = LatentDirichletAllocation(n_topics=5, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(corpus)

    components_5 = np.argsort(lda_5.components_)[::1]
    components_10 = np.argsort(lda_10.components_)[::1]
    print("Top 10 word for 5 topic model")

    for i, item in enumerate(components_5):
        words = []
        for j in range(0,10):
            words += features[components_5[i,j]]
        print(words)


    print("Top 10 word for 10 topic model")
    for i, item in enumerate(components_10):
        words = []
        for j in range(0,10):
            words += features[components_10[i,j]]
        print(words)
    for i in range(2,11):
        lda = LatentDirichletAllocation(n_topics=i, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(corpus)
        print("Perplexity{}:{}".format(i,lda.perplexity(corpus)))




if __name__ == '__main__':
    # train_models('movie-pang02.csv')
    train_topic_models('movie-pang02.csv')
