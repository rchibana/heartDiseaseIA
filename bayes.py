# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.naive_bayes import GaussianNB

from classifier import Classifier


class TreeClassifier(Classifier):

    def __init__(self):
        self.classifier = GaussianNB()

    def do_train(self, X, y):
        self.classifier.fit(X, y)

    def do_classification(self, X, y):
        self.classifier.predict(X, y)
