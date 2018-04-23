# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:54:50 2018

@author: mgungor
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score

models = [('MultiNB', MultinomialNB(alpha=0.03)),
          ('Calibrated MultiNB', CalibratedClassifierCV(
              MultinomialNB(alpha=0.03), method='isotonic')),
          ('Calibrated BernoulliNB', CalibratedClassifierCV(
              BernoulliNB(alpha=0.03), method='isotonic')),
          ('Calibrated Huber', CalibratedClassifierCV(
              SGDClassifier(loss='modified_huber', alpha=1e-4,
                            max_iter=10000, tol=1e-4), method='sigmoid')),
          ('Logit', LogisticRegression(C=30))]
test_author = pd.read_csv("test_author.txt", header=None, names=["author"])
train = pd.read_csv('train.csv')
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,3,3,1,1])
X_train = vectorizer.fit_transform(train.text.values)
authors = ['MWS','EAP','HPL']
y_train = train.author.apply(authors.index).values
yvalid = test_author.author.apply(authors.index).values
clf.fit(X_train, y_train)

test = pd.read_csv('test.csv', index_col=0)
X_test = vectorizer.transform(test.text.values)
results = clf.predict(X_test)

print("f1 Score")
print(f1_score(yvalid, results, average='macro'))
print("f1 Score Individual")
print(f1_score(yvalid, results, average=None))
print("Accuracy")
print(accuracy_score(yvalid, results))