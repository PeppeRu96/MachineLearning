import os
import numpy as np
import scipy.special

import evaluation.common as eval

SCRIPT_PATH = os.path.dirname(__file__)
COMMEDIA_LL_PATH = os.path.join("..", "Data", "commedia_ll.npy")
COMMEDIA_LABELS_PATH = os.path.join("..", "Data", "commedia_labels.npy")

COMMEDIA_LL_EPS1_PATH = os.path.join("..", "Data", "commedia_ll_eps1.npy")
COMMEDIA_LABELS_EPS1_PATH = os.path.join("..", "Data", "commedia_labels_eps1.npy")

if __name__ == "__main__":
    ll_commedia = np.load(COMMEDIA_LL_PATH)
    labels_commedia = np.load(COMMEDIA_LABELS_PATH)

    ll_commedia_eps1 = np.load(COMMEDIA_LL_EPS1_PATH)
    labels_commedia_eps1 = np.load(COMMEDIA_LABELS_EPS1_PATH)

    C = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    vPriors = np.array([0.3, 0.4, 0.3]).reshape(3, 1)

    print("EPS: 0.001")
    # print("Log likelihoods shape: ", log_likelihoods.shape)
    S_joint = ll_commedia * vPriors
    # print("Joint S matrix shape: ", S_joint.shape)
    marginal = scipy.special.logsumexp(S_joint, axis=0)
    S_posterior = S_joint - marginal
    #pred_labels = np.argmax(S_posterior, 0)

    Costs = C @ S_posterior
    predictions = np.argmin(Costs, axis=0)

    # Calculating DCF
    print("Confusion matrix with eps 0.001:")
    M = eval.get_confusion_matrix(predictions, labels_commedia, show=True)
    print("")

    DCFu = eval.bayes_multiclass_dcfu(M, vPriors, C)
    DCF = eval.bayes_multiclass_dcf(M, vPriors, C)

    print("DCFu: ", DCFu)
    print("DCF: ", DCF)
    print("")


    # EPS = 1
    print("EPS: 1")
    S_joint = ll_commedia_eps1 * vPriors
    marginal = scipy.special.logsumexp(S_joint, axis=0)
    S_posterior = S_joint - marginal
    Costs = C @ S_posterior
    predictions = np.argmin(Costs, axis=0)

    # Calculating DCF
    print("Confusion matrix with eps 1:")
    M = eval.get_confusion_matrix(predictions, labels_commedia_eps1, show=True)
    print("")

    DCFu = eval.bayes_multiclass_dcfu(M, vPriors, C)
    DCF = eval.bayes_multiclass_dcf(M, vPriors, C)

    print("DCFu: ", DCFu)
    print("DCF: ", DCF)
    print("")

    # UNIFORM PRIORS AND UNIFORM COSTS
    C = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    vPriors = np.array([1/3, 1/3, 1/3]).reshape(3, 1)

    # EPS: 0.001
    print("Uniform priors and costs with eps: 0.001..")

    S_joint = ll_commedia * vPriors
    marginal = scipy.special.logsumexp(S_joint, axis=0)
    S_posterior = S_joint - marginal
    Costs = C @ S_posterior
    pred_labels = np.argmax(S_posterior, 0)
    predictions = np.argmin(Costs, axis=0)

    # Calculating DCF
    print("Confusion matrix with eps 0.001:")
    M = eval.get_confusion_matrix(pred_labels, labels_commedia, show=True)
    print("")

    DCFu = eval.bayes_multiclass_dcfu(M, vPriors, C)
    DCF = eval.bayes_multiclass_dcf(M, vPriors, C)

    print("DCFu: ", DCFu)
    print("DCF: ", DCF)
    print("")

    # EPS: 1
    print("Uniform priors and costs with eps: 1..")

    S_joint = ll_commedia_eps1 * vPriors
    marginal = scipy.special.logsumexp(S_joint, axis=0)
    S_posterior = S_joint - marginal
    Costs = C @ S_posterior
    predictions = np.argmin(Costs, axis=0)

    # Calculating DCF
    print("Confusion matrix with eps 1:")
    M = eval.get_confusion_matrix(predictions, labels_commedia_eps1, show=True)
    print("")

    DCFu = eval.bayes_multiclass_dcfu(M, vPriors, C)
    DCF = eval.bayes_multiclass_dcf(M, vPriors, C)

    print("DCFu: ", DCFu)
    print("DCF: ", DCF)
    print("")
