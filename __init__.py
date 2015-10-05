__author__ = 'rchibana'

from utils import get_data, encode_target
from tree import TreeClassifier
from bayes import GaussianNB
from neural_network import NeuralNetworkClassifier

# Return the instance to local test
def instance_of_classifier():

    # return TreeClassifier(min_samples_split=20, random_state=99)
    # return GaussianNB(min_samples_split=20, random_state=99)
    return NeuralNetworkClassifier(min_samples_split=20, random_state=99)

if __name__ == '__main__':

    FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
                "slope", "ca", "thal"]

    data = get_data()

    data, targets = encode_target(data, "num")
    y = data["Target"]
    X = data[FEATURES]

    try:
        new_classifier = instance_of_classifier()
        new_classifier.do_train(X, y)
        prediction = new_classifier.do_classification(X, y)

        score = new_classifier.do_cross_validation(new_classifier.classifier, X, y)
    except Exception, e:
        print e

    print("mean: {:.3f} (std: {:.3f})".format(score.mean(), score.std()))
