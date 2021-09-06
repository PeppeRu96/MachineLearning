import sklearn.datasets

import preproc.dstools as dst
from classifiers.logistic_regression import LogisticRegressionClassifier


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L

if __name__ == "__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = dst.split_db_2to1(D, L, 2.0/3.0)
    lambdas = [0, 1e-6, 1e-3, 1.0]

    print("LINEAR LR")
    for l in lambdas:
        print("Lambda: ", l)
        lr_model = LogisticRegressionClassifier(2)
        lr_model.train(DTR, LTR, l, verbose=1)
        pred_labels = lr_model.inference(DTE)

        correct_predictions = (pred_labels == LTE).sum()
        accuracy = correct_predictions / pred_labels.shape[0]
        print("Accuracy: ", accuracy)
        print("Error rate: ", 1-accuracy)
        print("\n")
    print("\n")
    print("QUADRATIC LR")
    for l in lambdas:
        print("Lambda: ", l)
        lr_model = LogisticRegressionClassifier(2)
        lr_model.train(DTR, LTR, l, expand_feature_space_func=LogisticRegressionClassifier.quadratic_feature_expansion, verbose=1)
        pred_labels = lr_model.inference(DTE)

        correct_predictions = (pred_labels == LTE).sum()
        accuracy = correct_predictions / pred_labels.shape[0]
        print("Accuracy: ", accuracy)
        print("Error rate: ", 1-accuracy)
        print("\n")