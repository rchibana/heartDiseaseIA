# -*- coding: utf-8 -*-

import subprocess

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from classifier import Classifier


class TreeClassifier(Classifier):

    def __init__(self, min_samples_split=20, random_state=99):
        self.classifier = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                                 random_state=random_state)

    def do_train(self, X, y):
        self.classifier.fit(X, y)

    def do_classification(self, X, y):
        self.classifier.predict(X[:, 'age':'thal'])
        print('wtf')

    def visualize_tree(tree, feature_names):
        """Create tree png using graphviz.

        Args
        ----
        tree -- scikit-learn DecsisionTree.
        feature_names -- list of feature names.
        """
        with open("dt.dot", 'w') as f:
            export_graphviz(tree, out_file=f, feature_names=feature_names)

        command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
        try:
            subprocess.check_call(command)
        except Exception, e:
            print(e)
            exit("Could not run dot, ie graphviz, to produce visualization")
