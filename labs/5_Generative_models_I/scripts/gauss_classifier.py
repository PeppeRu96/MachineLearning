import numpy as np
import sklearn.datasets
import preproc.dstools as dst
from classifiers.gaussian_classifier import MVG_Classifier

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

    mvg = MVG_Classifier()

    mvg.train(DTR, LTR)


    # Perform gaussian classification on test set
    # Likelihood standard
    print("Evaluating MVG classifier against test dataset (Test samples: %d) with standard MVG densities.." % DTE.shape[1])
    pred_labels1 = mvg.inference(DTE, Pc, False)
    cnt = (pred_labels1 == LTE).sum()
    acc = cnt / LTE.shape[0]
    err = 1 - acc
    print("Accuracy: ", acc)
    print("Error: ", err)
    print(" ")

    # Log-likelihood
    print("Evaluating MVG classifier against test dataset (Test samples: %d) with logarithm MVG densities.." % DTE.shape[1])
    pred_labels2 = mvg.inference(DTE, Pc, True)
    cnt = (pred_labels2 == LTE).sum()
    acc = cnt / LTE.shape[0]
    err = 1 - acc
    print("Accuracy: ", acc)
    print("Error: ", err)
    print(" ")

    # Cross-validation test
    K = D.shape[1]
    print("Evaluating MVG with K FOLD cross-validation with K=%d" % K)
    folds, folds_labels = dst.kfold_split(D, L, K)
    samples = folds.shape[2]*folds.shape[0]
    print("Total samples: ", samples)
    correct = 0

    for DTR, LTR, DTE, LTE in dst.kfold_generate(folds, folds_labels):
        mvg = MVG_Classifier()
        mvg.train(DTR, LTR)
        pred_labels = mvg.inference(DTE, Pc, True)
        cnt = (pred_labels == LTE).sum()
        correct = correct + cnt
        #acc = cnt / LTE.shape[0]
        #err = 1 - acc
        #print("%d: Accuracy: " % (i), acc)
        #print("%d: Error: " % (i), err)

    acc = correct / samples
    err = 1 - acc
    print(" ")
    print("Total Accuracy: ", acc)
    print("Total Error: ", err)
