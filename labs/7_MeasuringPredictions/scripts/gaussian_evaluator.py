import numpy as np
import sklearn.datasets
import preproc.dstools as dst
from classifiers.gaussian_classifier import MVG_Classifier
import evaluation.common as eval

# Prior distributions
Pc = np.array([1 / 3, 1 / 3, 1 / 3]).reshape(3, 1)

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

if (__name__ == "__main__"):
    print("Loading IRIS dataset..")
    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = dst.split_db_2to1(D, L, 2.0/3.0)

    print("Dataset shape: ", D.shape)
    print("Labels shape: ", L.shape)
    print(" ")

    # Confusion matrix of MVG
    mvg = MVG_Classifier()
    mvg.train(DTR, LTR)
    pred_labels = mvg.inference(DTE, Pc)
    print("Confusion Matrix of MVG:")
    eval.get_confusion_matrix(pred_labels, LTE, show=True)
    print("")

    # Confusion matrix of Tied MVG
    tied_gauss = MVG_Classifier()
    tied_gauss.train(DTR, LTR, naive=False, tied=True)
    pred_labels = tied_gauss.inference(DTE, Pc)
    print("Confusion matrix of Tied MVG:")
    eval.get_confusion_matrix(pred_labels, LTE, show=True)
    print("")
