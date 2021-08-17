import numpy as np

from preproc.dstools import DEBUG
import classifiers.distributions.gaussian as gd
import scipy.special

class MVG_Classifier:

    def __init__(self):
        self.DTR = None
        self.LTR = None
        self.mu = None
        self.C = None

    def train(self, DTR, LTR, verbose=0):
        self.DTR = DTR
        self.LTR = LTR

        mu, C = MVG_classifier_train(DTR, LTR, verbose)
        self.mu = mu
        self.C = C

        return mu, C

    def inference(self, D, Pc, use_log=True):
        return MVG_classifier_inference(D, self.mu, self.C, Pc, use_log)

# Train a multivariate gaussian classifier given a train dataset DTR-LTR
# It returns a mu np array with shape (#dimensions, #classes) where each column i contains the mean for the class i
# and a C np array with shape (#classes, #dimensions, #dimensions) containing the covariance matrices for each class
def MVG_classifier_train(DTR, LTR, verbose=0):
    if verbose:
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
def MVG_classifier_inference(D, mu, C, Pc, use_log=True):
    # Compute the likelihoods fX|C = N(xt|muc, Sigmac)
    if DEBUG:
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