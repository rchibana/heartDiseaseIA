__author__ = 'rchibana'

from utils import get_data, encode_target
from tree import TreeClassifier

if __name__ == '__main__':

    FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
                "slope", "ca", "thal"]

    data = get_data()

    data, targets = encode_target(data, "num")
    y = data["Target"]
    X = data[FEATURES]

    try:
        tree_classifier = TreeClassifier(min_samples_split=20, random_state=99)
        tree_classifier.do_train(X, y)
        prediction = tree_classifier.do_classification(X, y)

        score = tree_classifier.do_cross_validation(tree_classifier.classifier, X, y)
    except Exception, e:
        print e

    print("mean: {:.3f} (std: {:.3f})".format(score.mean(), score.std()))
