__author__ = 'rchibana'

from sklearn.cross_validation import  cross_val_score


class Classifier(object):

    def do_classification(self, X, y):
        pass

    def do_train(self, X, y):
        pass

    def do_cross_validation(self, dt, X, y):
        return cross_val_score(dt, X, y, cv=10)
