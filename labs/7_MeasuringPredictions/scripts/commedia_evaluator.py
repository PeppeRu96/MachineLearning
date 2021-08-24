import os
import numpy as np
import scipy.special

import evaluation.common as eval

SCRIPT_PATH = os.path.dirname(__file__)
COMMEDIA_LL_PATH = os.path.join("..", "Data", "commedia_ll.npy")
COMMEDIA_LABELS_PATH = os.path.join("..", "Data", "commedia_labels.npy")

def discrete_lienar_classifier_inference(D, pi, Pc):
    log_likelihoods = np.log(pi) @ D
    #print("Log likelihoods shape: ", log_likelihoods.shape)
    Pc = np.array(Pc).reshape(3, 1)
    S_joint = log_likelihoods * Pc
    #print("Joint S matrix shape: ", S_joint.shape)
    marginal = scipy.special.logsumexp(S_joint, axis=0)
    S_posterior = S_joint - marginal
    pred_labels = np.argmax(S_posterior, 0)

    return pred_labels


if __name__ == '__main__':
    ground_truths = np.load(COMMEDIA_LABELS_PATH)
    print("Ground truths shape: ", ground_truths.shape)
    Pc = [1/3, 1/3, 1/3]
    log_likelihoods = np.load(COMMEDIA_LL_PATH)
    print("Likelihoods shape: ", log_likelihoods.shape)

    Pc = np.array(Pc).reshape(3, 1)

    # Inference
    S_joint = log_likelihoods * Pc
    marginal = scipy.special.logsumexp(S_joint, axis=0)
    S_posterior = S_joint - marginal
    pred_labels = np.argmax(S_posterior, 0)

    print("Confusion Matrix for commedia tercet classifier:")
    eval.get_confusion_matrix(pred_labels, ground_truths, show=True)

