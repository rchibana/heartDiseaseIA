# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.neural_network import BernoulliRBM

from classifier import Classifier


class NeuralNetworkClassifier(Classifier):

    def __init__(self):
        self.classifier = BernoulliRBM()

    def do_train(self, X, y):
        self.classifier.fit(X, y)

    def do_classification(self, X, y):
        self.classifier.predict(X, y)
