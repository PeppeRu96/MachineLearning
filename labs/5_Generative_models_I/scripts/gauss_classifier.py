from random import randrange

import numpy as np
import sklearn.datasets
import sys
import scipy.special

sys.path.append('../../2_Iris/scripts/')
sys.path.append('../../4_Probability_Density/scripts/')

import gauss_density as gd

# Prior probability
Pc = np.array([1 / 3, 1 / 3, 1 / 3]).reshape(3, 1)


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def split_db_2to1(D, L, train_fraction, seed=0):
    print("Splitting dataset in %f train data and %f test data" % (train_fraction, (1-train_fraction)))
    nTrain = int(D.shape[1] * train_fraction)
    print("Train samples: %d; Test samples: %d" % (nTrain, D.shape[1]-nTrain))
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def cross_validation_split(dataset, labels, folds=3):
    print("Cross-validation split with K=%d" %(folds))
    dataset_split = list()
    dataset_copy = list(dataset.T)
    labels_split = list()
    fold_size = int(len(dataset_copy) / folds)
    for i in range(folds):
        fold = list()
        fold_labels = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            fold_labels.append(labels[index])
        fold = list(np.array(fold).T)
        dataset_split.append(fold)
        labels_split.append(fold_labels)

    dataset_split = np.array(dataset_split)
    return dataset_split, np.array(labels_split)

def train_gaussian_classifier(DTR, LTR):
    print("Training Gaussian Classifier on %d training samples.." % (DTR.shape[1]))
    # Calculating empirical mean and variance for each class which correspond to the ML estimated
    mu = []
    C = []
    nc = len(set(LTR))
    for i in range(nc):
        DTRi = DTR[:, (LTR == i)]
        mu_i = DTRi.mean(1)
        mu_i = mu_i.reshape(mu_i.size, 1)
        mu.append(mu_i)

        DTRCi = DTRi - mu_i
        Ci = (np.dot(DTRCi, DTRCi.T)) / DTRCi.shape[1]
        C.append(Ci)

    mu = np.array(mu)
    C = np.array(C)
    return mu, C

def gaussian_classifier(D, mu, C, Pc, use_log=True):

    # Compute the likelihoods fX|C = N(xt|muc, Sigmac)
    if (use_log):
        print("Gaussian Classification with log-Likelihood:")
    else:
        print("Gaussian Classification with Likelihood standard (non-logarithm):")
    S = []
    for i in range(mu.shape[0]):
        if (use_log):
            # log-likelihood
            log_likehood_i = gd.logpdf_GAU_ND(D, mu[i], C[i])
            lc = log_likehood_i + np.log(Pc[i])
            S.append(lc)
        else:
            # Likelihood
            Si = np.exp(gd.logpdf_GAU_ND(D, mu[i], C[i]))
            # Joint probability
            Si = Si * Pc[i]
            S.append(Si)
    SJoint = np.array(S)
    marginal = None
    SPost = None
    if (use_log):
        marginal = scipy.special.logsumexp(SJoint, axis=0)
        SPost = SJoint - marginal
    else:
        marginal = SJoint.sum(axis=0)
        SPost = SJoint / marginal

    pred_labels = np.argmax(SPost, 0)

    return pred_labels

def print_dimensions(DTR, DTE, LTR, LTE):
    print("DTR shape: ", DTR.shape)
    print("LTR shape: ", LTR.shape)
    print("DTE shape: ", DTE.shape)
    print("LTE shape: ", LTE.shape)

if (__name__ == "__main__"):
    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L, 2.0/3.0)

    print("Dataset shape: ", D.shape)
    print("Labels shape: ", L.shape)
    #print_dimensions(DTR, DTE, LTR, LTE)

    mu, C = train_gaussian_classifier(DTR, LTR)

    # Perform gaussian classification on test set
    # Likelihood standard
    pred_labels1 = gaussian_classifier(DTE, mu, C, Pc, False)
    cnt = (pred_labels1 == LTE).sum()
    acc = cnt / LTE.shape[0]
    err = 1 - acc
    print("Accuracy: ", acc)
    print("Error: ", err)

    # Log-likelihood
    pred_labels2 = gaussian_classifier(DTE, mu, C, Pc, True)

    cnt = (pred_labels2 == LTE).sum()
    acc = cnt / LTE.shape[0]
    err = 1 - acc
    print("Accuracy: ", acc)
    print("Error: ", err)

    # Cross-validation test
    folds, folds_labels = cross_validation_split(D, L, 50)
    #folds, folds_labels = cross_validation_split(D, L, D.shape[1])
    samples = folds.shape[2]*folds.shape[0]
    print("Total samples: ", samples)
    correct = 0
    for i in range(folds.shape[0]):
        print("--------- Cross-validation Iteration %d ----------" % (i))
        folds_copy = list(folds)
        folds_labels_copy = list(folds_labels)

        DTE = np.array(folds_copy.pop(i))
        LTE = np.array(folds_labels_copy.pop(i))

        folds_copy = np.array(folds_copy)
        folds_labels_copy = np.array(folds_labels_copy)
        DTR = folds_copy[0]
        for j in range(1, folds_copy.shape[0]):
            DTR = np.concatenate((DTR, folds_copy[j]), axis=1)

        LTR = folds_labels_copy.flatten()
        #print_dimensions(DTR, DTE, LTR, LTE)

        mu, C = train_gaussian_classifier(DTR, LTR)
        pred_labels = gaussian_classifier(DTE, mu, C, Pc, False)
        cnt = (pred_labels == LTE).sum()
        correct = correct + cnt
        acc = cnt / LTE.shape[0]
        err = 1 - acc
        print("%d: Accuracy: " % (i), acc)
        print("%d: Error: " % (i), err)

    acc = correct / samples
    err = 1 - acc
    print("Total Accuracy: ", acc)
    print("Total Error: ", err)