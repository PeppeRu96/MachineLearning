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
    mvg.evaluate(DTE, LTE, Pc, use_log=False)

    # Log-likelihood
    print("Evaluating MVG classifier against test dataset (Test samples: %d) with logarithm MVG densities.." % DTE.shape[1])
    mvg.evaluate(DTE, LTE, Pc)

    # Naive Bayes Gaussian
    print("Evaluating Naive Bayes Gaussian classifier against test dataset (Test samples: %d) with logarithm densities.." % DTE.shape[1])
    naive_gauss = MVG_Classifier()
    naive_gauss.train(DTR, LTR, naive=True, verbose=True)
    naive_gauss.evaluate(DTE, LTE, Pc)

    # Tied Gaussian
    print("Evaluating Tied Gaussian classifier against test dataset (Test samples: %d) with logarithm densities.." %DTE.shape[1])
    tied_gauss = MVG_Classifier()
    tied_gauss.train(DTR, LTR, naive=False, tied=True, verbose=True)
    tied_gauss.evaluate(DTE, LTE, Pc)

    # Naive Tied Gaussian
    print("Evaluating Naive Tied Gaussian classifier against test dataset (Test samples: %d) with logarithm densities.." % DTE.shape[1])
    naive_tied_gauss = MVG_Classifier()
    naive_tied_gauss.train(DTR, LTR, naive=True, tied=True, verbose=True)
    naive_tied_gauss.evaluate(DTE, LTE, Pc)

    # Cross-validation test
    K = D.shape[1]
    mvg.evaluate(D, L, Pc, Kfold=True, K=K)
    naive_gauss.evaluate(D, L, Pc, Kfold=True, K=K)
    tied_gauss.evaluate(D, L, Pc, Kfold=True, K=K)
    naive_tied_gauss.evaluate(D, L, Pc, Kfold=True, K=K)
