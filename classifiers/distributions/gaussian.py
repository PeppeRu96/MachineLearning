import numpy as np
import scipy.special

import json


def GAU_pdf(x, mu, var):
    y = np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
    return y

# Computes the logarithm of the gaussian distribution density of a numpy array x, given the mean mu and the variance var
def GAU_logpdf(x, mu, var):
    y = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - ((x - mu) ** 2) / (2 * var)
    return y

# Computes the logarithm of the multivariate gaussian distribution density of a np array x with shape (#dimensions, #samples)
# given the mean mu (#dimensions, 1) and the covariance matrix C (#dimensions, #dimensions)
c1 = - 0.5 * np.log(2 * np.pi)
def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    k1 = c1 * M
    k2 = - 0.5 * np.linalg.slogdet(C)[1]
    Cinv = np.linalg.inv(C)
    X_c = x - mu
    Y = k1 + k2 - 0.5 * np.diag((X_c.T @ Cinv @ X_c))
    return Y

def logpdf_GMM(X, gmm):
    S = []
    for (w, mu, sigma) in gmm:
        log_class_conditional_density_g = logpdf_GAU_ND(X, mu, sigma)
        joint_log_density_g = log_class_conditional_density_g + np.log(w)
        S.append(joint_log_density_g)
    S = np.array(S)
    gmm_log_density = scipy.special.logsumexp(S, axis=0)

    return gmm_log_density

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]
