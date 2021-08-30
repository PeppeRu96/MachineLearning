import numpy as np
import sklearn.datasets
import preproc.dstools as dst
from classifiers.gmm_classifier import GMM_Classifier

import density_estimation.gaussian_mixture_model as gmm

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

    classes = len(set(LTR))
    gmm_all = []
    gmm_diagonal_all = []
    gmm_tied_all = []
    for i in range(classes):
        DTRi = DTR[:, (LTR==i)]
        gmm_lbg_i_all = gmm.LBG_estimate(DTRi, 0.1, stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == 16), verbose=0)
        gmm_all.append(gmm_lbg_i_all)

        gmm_lbg_i_diagonal_all = gmm.LBG_estimate(DTRi, 0.1, diag_cov=True, stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == 16), verbose=0)
        gmm_diagonal_all.append(gmm_lbg_i_diagonal_all)

        gmm_lbg_i_tied_all = gmm.LBG_estimate(DTRi, 0.1, tied_cov=True, stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == 16), verbose=0)
        gmm_tied_all.append(gmm_lbg_i_tied_all)

    # For each gmm generated from the LBG algorithm..
    print("{:>30}".format("GMM Type"), end='')
    for gmm in gmm_all[0]:
        print("{:>10}".format(len(gmm)), end='')
    print("")
    config_cnt = len(gmm_all[0])
    print("{:>30}".format("Full covariance"), end='')
    for conf_i in range(config_cnt):
        gmms_i = []
        for i in range(classes):
            gmms_i.append(gmm_all[i][conf_i])
        # Standard
        gmm_classifier = GMM_Classifier(gmms_i)
        pred_labels = gmm_classifier.inference(DTE, Pc)
        correct = (pred_labels == LTE).sum()
        errorRate = 1 - correct/pred_labels.shape[0]
        print("{:>10.1f}%".format(errorRate * 100), end='')
    print("")

    print("{:>30}".format("Diag covariance"), end='')
    for conf_i in range(config_cnt):
        gmms_diag_i = []
        for i in range(classes):
            gmms_diag_i.append(gmm_diagonal_all[i][conf_i])
        # Standard
        gmm_classifier = GMM_Classifier(gmms_diag_i)
        pred_labels = gmm_classifier.inference(DTE, Pc)
        correct = (pred_labels == LTE).sum()
        errorRate = 1 - correct/pred_labels.shape[0]
        print("{:>10.1f}%".format(errorRate * 100), end='')
    print("")

    print("{:>30}".format("Tied covariance"), end='')
    for conf_i in range(config_cnt):
        gmms_tied_i = []
        for i in range(classes):
            gmms_tied_i.append(gmm_tied_all[i][conf_i])
        # Standard
        gmm_classifier = GMM_Classifier(gmms_tied_i)
        pred_labels = gmm_classifier.inference(DTE, Pc)
        correct = (pred_labels == LTE).sum()
        errorRate = 1 - correct/pred_labels.shape[0]
        print("{:>10.1f}%".format(errorRate * 100), end='')
    print("")
