import numpy as np
import scipy.special

import classifiers.distributions.gaussian as gau
import preproc.dstools as dst
from density_estimation.gaussian_mixture_model import LBG_estimate

class GMM_Classifier:

    def __init__(self, gmms):
        self.gmms = gmms

    def inference(self, D, Pc):
        S = []
        for i in range(len(self.gmms)):
            log_likelihood_i = gau.logpdf_GMM(D, self.gmms[i])
            lc = log_likelihood_i + np.log(Pc[i])
            S.append(lc)
        Sjoint = np.array(S)
        marginal = scipy.special.logsumexp(Sjoint, axis=0)
        SPost = Sjoint - marginal

        pred_labels = np.argmax(SPost, 0)

        return pred_labels

    def compute_binary_llr(self, D):
        return (gau.logpdf_GMM(D, self.gmms[1]) - gau.logpdf_GMM(D, self.gmms[0])).flatten()

def cross_validate_gmm(folds_data, folds_labels, preproc_conf, alpha, psi, diag_cov, tied_cov, max_components, verbose=False):
    """
    Cross-validate log2(max_components) different GMM classifiers (1 comp, 2 comps, 4 comps, 8 comps, etc..) using
    LBG estimation and a K-fold cross-validation for a given preprocess configuration
    and a given model configuration (tied, diag, etc.).
    :return a tuple (scores, labels) where scores and labels are two nparray of shape (#classifiers, #validation_samples)
            where #classifiers = log2(max_components) + 1 and #validation_samples = #folds * fold_size
    """
    if verbose:
        diag_cov_str = "Diagonal" if diag_cov else ""
        tied_cov_str = "Tied" if tied_cov else ""
        print("\t5-Fold Cross-Validation {} {} GMM (components from 1 to {}) - Preprocessing: {}".format(
            diag_cov_str, tied_cov_str, max_components, preproc_conf))
    iterations = 1
    K = folds_data.shape[0]
    nk = folds_labels.shape[1]
    scores = np.zeros((max_components, K*nk))
    labels = np.zeros((max_components, K*nk))
    k = 0
    for DTR, LTR, DTE, LTE in dst.kfold_generate(folds_data, folds_labels):
        # Preprocess data
        DTR, DTE = preproc_conf.apply_preproc_pipeline(DTR, LTR, DTE)

        # Train all the gmms from 1 component to max_components components for class 0 and 1
        DTR0 = DTR[:, (LTR == 0)]
        DTR1 = DTR[:, (LTR == 1)]
        gmms_h0 = LBG_estimate(DTR0, alpha, psi=psi, diag_cov=diag_cov, tied_cov=tied_cov,
                                   stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == max_components), verbose=0)
        gmms_h1 = LBG_estimate(DTR1, alpha, psi=psi, diag_cov=diag_cov, tied_cov=tied_cov,
                                   stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == max_components), verbose=0)
        gmm_classifiers = []
        for g0, g1 in zip(gmms_h0, gmms_h1):
            gmms = [g0, g1]
            gmm_classifiers.append(GMM_Classifier(gmms))

        # Now gmm_classifiers contains log2(max_components) + 1 gmm classifiers

        # Validate
        for i, gmm_classifier in enumerate(gmm_classifiers):
            s = gmm_classifier.compute_binary_llr(DTE)
            scores[i, k*nk:(k+1)*nk] = s
            labels[i, k*nk:(k+1)*nk] = LTE

        k += 1
        iterations += 1

    return scores, labels