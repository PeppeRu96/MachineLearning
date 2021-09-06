import numpy as np
import scipy.special

from preproc.dstools import DEBUG
import classifiers.distributions.gaussian as gd
import preproc.dstools as dst

class MVG_Classifier:

    def __init__(self, K):
        self.DTR = None
        self.LTR = None
        self.K = K
        self.mu = None
        self.C = None
        self.naive = False
        self.tied = False

    def train(self, DTR, LTR, naive=False, tied=False, verbose=0):
        self.DTR = DTR
        self.LTR = LTR

        mu, C = MVG_classifier_train(DTR, LTR, self.K, naive, tied, verbose)
        self.mu = mu
        self.C = C
        self.naive = naive
        self.tied = tied

        return mu, C

    def inference(self, D, Pc, use_log=True):
        return MVG_classifier_inference(D, self.mu, self.C, Pc, self.naive, self.tied, use_log)

    def compute_binary_classifier_llr(self, D):
        return MVG_binary_classifier_llr(D, self.mu, self.C, self.tied)

    def evaluate(self, D, L, Pc, Kfold=False, K=None, use_log=True):
        if (Kfold == False):
            pred_labels = self.inference(D, Pc, use_log)
            cnt = (pred_labels == L).sum()
            acc = cnt / L.shape[0]
            err = 1 - acc
            print("Accuracy: ", acc)
            print("Error: ", err)
            print(" ")
        else:
            naive_str = ""
            tied_str = ""
            if self.naive:
                naive_str = "Naive"
            if self.tied:
                tied_str = "Tied"
            print("------------------")
            print("Evaluating %s %s MVG with K FOLD cross-validation with K=%d" % (naive_str, tied_str, K))
            folds, folds_labels = dst.kfold_split(D, L, K)
            samples = folds.shape[2] * folds.shape[0]
            print("Total samples: ", samples)
            correct = 0

            for DTR, LTR, DTE, LTE in dst.kfold_generate(folds, folds_labels):
                mvg = MVG_Classifier(self.K)
                mvg.train(DTR, LTR, self.naive, self.tied)
                pred_labels = mvg.inference(DTE, Pc)
                cnt = (pred_labels == LTE).sum()
                correct = correct + cnt
                # acc = cnt / LTE.shape[0]
                # err = 1 - acc
                # print("%d: Accuracy: " % (i), acc)
                # print("%d: Error: " % (i), err)

            acc = correct / samples
            err = 1 - acc
            print(" ")
            print("Total Accuracy: ", acc)
            print("Total Error: ", err)
            print("------------------")
            print("")

# Train a multivariate gaussian classifier given a train dataset DTR-LTR
# It returns a mu np array with shape (#dimensions, #classes) where each column i contains the mean for the class i
# and a C np array with shape (#classes, #dimensions, #dimensions) containing the covariance matrices for each class
def MVG_classifier_train(DTR, LTR, K, naive=False, tied=False, verbose=0):
    if verbose:
        naive_str = ""
        tied_str = ""
        if naive:
            naive_str = "Naive"
        if (tied):
            tied_str = "Tied"
        print("Training %s %s Gaussian Classifier on %d training samples.." % (naive_str, tied_str, DTR.shape[1]))
    # Calculating empirical mean and variance for each class which correspond to the ML estimated
    mu = []
    C = []
    nc = K
    for i in range(nc):
        DTRi = DTR[:, (LTR == i)]
        mu_i = DTRi.mean(1)
        mu_i = mu_i.reshape(mu_i.size, 1)
        mu.append(mu_i)

        DTRCi = DTRi - mu_i
        Ci = np.dot(DTRCi, DTRCi.T)
        if naive:
            Ci = Ci * np.eye(DTRi.shape[0])

        if tied:
            if i == 0:
                C = Ci
            else:
                C = C + Ci
        else:
            Ci = Ci / DTRCi.shape[1]
            C.append(Ci)

    mu = np.array(mu)
    if tied:
        C = C / DTR.shape[1]
    else:
        C = np.array(C)

    if (DEBUG):
        print("mu matrix:")
        print(mu)
        print("\nCovariance matrices:")
        print(C)

    return mu, C

# Perform the inference of the class label using a trained MVG classifier on a dataset D (#dimensions, #samples)
# Pc: vector of prior probabilities
# use_log: use log densities for avoiding numerical problems
# Returns a vector containing the predicted labels (from 0 to K-1) where K is the number of classes
def MVG_classifier_inference(D, mu, C, Pc, naive=False, tied=False, use_log=True):
    # Compute the likelihoods fX|C = N(xt|muc, Sigmac)
    if DEBUG:
        if (use_log):
            print("Gaussian Classification with log-Likelihood:")
        else:
            print("Gaussian Classification with Likelihood standard (non-logarithm):")
    S = []
    for i in range(mu.shape[0]):
        C_curr = None
        if tied:
            C_curr = C
        else:
            C_curr = C[i]

        if (use_log):
            # log-likelihood
            log_likehood_i = gd.logpdf_GAU_ND(D, mu[i], C_curr)
            lc = log_likehood_i + np.log(Pc[i])
            S.append(lc)
        else:
            # Likelihood
            Si = np.exp(gd.logpdf_GAU_ND(D, mu[i], C_curr))
            # Joint distributions
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

# Computes the scores for a binary task and thus is application-agnostic
# The scores need to be compared with an appropriate threshold t in order to make the prediction
# Returns a np.array of scores
def MVG_binary_classifier_llr(D, mu, C, tied=False):
    mu0 = mu[0]
    mu1 = mu[1]
    if tied:
        C0 = C
        C1 = C
    else:
        C0 = C[0]
        C1 = C[1]

    # Compute log-likelihood ratio (score with probabilistic interpretation)
    ll0 = gd.logpdf_GAU_ND(D, mu0, C0)
    ll1 = gd.logpdf_GAU_ND(D, mu1, C1)

    llr_scores = ll1 - ll0

    return llr_scores