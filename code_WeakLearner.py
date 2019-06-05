#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:58:19 2018

@author: 
"""
import re
from textblob import TextBlob

def clean_tweet(tweet):
    ''' to remove links and special characters '''
    return ' '.join(re.sub("(@[@A-Za-z]+)|([^@A-Za-z \t])|(\w+:\/\/\S+)", " ", \
                           tweet).split())
def analyze_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

from __future__ import print_function
import os

import pandas as pd
import numpy as np
from collections import Counter

from features_ import tfidf_vec_, count_vec_
from scipy.sparse import hstack
import scipy 

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

cached = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves','you','your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'also', 'said', 'would', 'k']
stopwords = set(cached)

def remove_stopwords(t):
    return ' '.join(x if x.lower() not in stopwords else '' for x in str(t).split())

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train['text'] = map(remove_stopwords, train['text'])
test['text'] = map(remove_stopwords, test['text'])

train['sent'] = map(analyze_sentiment, train['text'])
test['sent'] = map(analyze_sentiment, test['text'])

train['nw'] = map(lambda x:len(clean_tweet(x).split()), train['text'])
test['nw'] = map(lambda x:len(clean_tweet(x).split()), test['text'])
train['handle'] = map(lambda x: 1 if x.count('@') >= 1 else 0, train['text'])
test['handle'] = map(lambda x:1 if x.count('@') >= 1 else 0, test['text'])
train['link'] = map(lambda x: 1 if 'http' in x else 0, train['text'])
test['link'] = map(lambda x:1 if 'http' in x else 0, test['text'])
train['isRetweet'] = map(lambda x:int(x), train['isRetweet'])
test['isRetweet'] = map(lambda x:int(x), test['isRetweet'])
train['hash'] = map(lambda x: 1 if x.count('#') >= 1 else 0, train['text'])
test['hash'] = map(lambda x:1 if x.count('#') >= 1 else 0, test['text'])
train['time_h'] = map(lambda x:int(x.split()[-1].split(':')[0]) if x!="" else -1, train['created'])
test['time_h'] = map(lambda x:int(x.split()[-1].split(':')[0]) if x!="" else -1, test['created'])    
train['inv'] = map(lambda x: 1 if x.count('"') >= 1 else 0, train['text'])
test['inv'] = map(lambda x: 1 if x.count('"') >= 1 else 0, test['text'])
train['thank'] = map(lambda x:1 if x.lower().count('crooked') >= 1 else 0, train['text'])
test['thank'] = map(lambda x:1 if x.lower().count('crooked') >= 1 else 0, test['text'])


fvars = ['handle', 'sent', 'thank']

#import seaborn as sns
#sns.set()
#d = pd.DataFrame(train['label'])
#d['x'] = map(lambda x: 1 if 'trump' in x.lower().split() else 0, train['text'])
#sns.swarmplot(y="x", x="label", data=d)

v1 = tfidf_vec_(train['text'])
v2 = count_vec_(train['text'])
x = v1.transform(train['text']).toarray()
xtest = v1.transform(test['text']).toarray()
y = train['label']

x = np.hstack((x, train[fvars].values))
xtest = np.hstack((xtest, test[fvars].values))

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.25, \
                                                  random_state=1000)

#clf1 = LogisticRegression(class_weight = 'balanced')
#clf1 = RidgeClassifier()
#clf1 = MultinomialNB()
#clf1 = RandomForestClassifier(100, random_state = 100)
#clf1 = AdaBoostClassifier(n_estimators = 40, algorithm='SAMME')
clf1 = GradientBoostingClassifier(n_estimators = 40, loss = 'exponential', random_state=100)
clf1.fit(xtrain, ytrain)
score = clf1.score(xvalid, yvalid)
print("Accuracy : ", score)

clf1.fit(x,y)
ypreds = clf1.predict(xtest)
ypreds.mean()

k = test[['id']].copy()
k['Label'] = ypreds
k.rename(columns = {'id' : 'ID'}, inplace=True)
k.to_csv('sub10.csv', index=None, header=True)