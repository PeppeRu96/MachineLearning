import numpy as np
import sklearn.datasets

import preproc.dstools as dst
from classifiers.logistic_regression import LogisticRegressionClassifier

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


if __name__ == "__main__":
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = dst.split_db_2to1(D, L, 2.0/3.0)
    K = len(set(LTR).union(set(LTE)))

    lambdas = [0, 1e-6, 1e-3, 1.0]
    for l in lambdas:
        print("Lambda: ", l)
        lr_model = LogisticRegressionClassifier(K)
        lr_model.train(DTR, LTR, l, verbose=1)
        pred_labels = lr_model.inference(DTE)

        correct_predictions = (pred_labels == LTE).sum()
        accuracy = correct_predictions / pred_labels.shape[0]
        print("Accuracy: ", accuracy)
        print("Error rate: ", 1 - accuracy)
        print("\n")