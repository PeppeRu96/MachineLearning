import numpy as np
import matplotlib.pyplot as plt


def get_confusion_matrix(predicted_labels, true_labels, all_labels=None, show=False):
    """Build and return the confusion matrix for the predicted labels

    :param predicted_labels: a np array (#samples,) containing the predicted labels
    :param true_labels: a np array (#samples,) containing the ground truths
    :param all_labels: an array containing all the labels just in case neither the true_labels or predicte_labels contain all the labels
    :return: Returns the confusion matrix
    """

    labels = all_labels
    if all_labels is None:
        labels = set(true_labels)
        extended = set(predicted_labels)
        labels = labels.union(extended)

    label_to_index = {l: i for i, l in enumerate(labels)}
    conf_matr = np.zeros((len(labels), len(labels)))
    for p, t in zip(predicted_labels, true_labels):
        p_i = label_to_index[p]
        t_i = label_to_index[t]
        conf_matr[p_i, t_i] = conf_matr[p_i, t_i] + 1

    if show:
        print("########################################")
        print("\t\t\tClass")
        print("\t\t\t", end="")
        for i in labels:
            print("%d\t" % i, end="")
        print("")
        print("\t\t---", end="")
        for i in labels:
            print("----", end="")
        print("")
        for i, l in enumerate(labels):
            if (i == len(labels) // 2):
                print("Pred\t%d |\t" % l, end="")
            else:
                print("\t\t%d |\t" % l, end="")
            for j in range(len(labels)):
                print("%d\t" % conf_matr[i, j], end="")
            print("")
        print("########################################")

    return conf_matr


def bayes_binary_optimal_classifier(llr, pi_1, Cfn, Cfp, threshold=None):
    """
    Computes the predictions starting from the likelihood ratios already made on some samples
    :param llr: log likelihood ratios corresponding to class-conditional probabilities
    :param pi_1: P(C=Ht) - that is the prior probability of the true hypothesis
    :param Cfn: Cost for a false negative prediction
    :param Cfp: Cost for a false positive prediction
    :return: A np array with the predictions (0 or 1)
    """
    if (llr.ndim > 1):
        llr = llr.flatten()
    if threshold is None:
        threshold = -np.log((pi_1 * Cfn) / ((1 - pi_1) * Cfp))
    predictions = [1 if l > threshold else 0 for l in llr]

    return predictions

def bayes_binary_dcfu(conf_matr, pi_1, Cfn, Cfp):
    """
    Computes the Unnormalized Detection Cost Function or Unnormalized Bayes Empirical Risk for a Binary Task on a validation/evaluation dataset
    :param conf_matr: confusion matrix
    :param pi_1: prior probability for the true hypothesis
    :param Cfn: Cost for a false negative prediction
    :param Cfp: Cost for a false positive prediction
    :return: the unnormalized DCF computed on the application (pi_1, Cfn, Cfp)
    """

    FNR = conf_matr[0, 1] / (conf_matr[0, 1] + conf_matr[1, 1])
    FPR = conf_matr[1, 0] / (conf_matr[1, 0] + conf_matr[0, 0])

    DCFu = pi_1 * Cfn * FNR + (1 - pi_1) * Cfp * FPR

    return DCFu

def bayes_binary_dcf(conf_matr, pi_1, Cfn, Cfp):
    """
    Computes the Detection Cost Function or Bayes Empirical Risk for a Binary Task on a validation/evaluation dataset
    :param conf_matr: confusion matrix
    :param pi_1: prior probability for the true hypothesis
    :param Cfn: Cost for a false negative prediction
    :param Cfp: Cost for a false positive prediction
    :return: the DCF computed on the application (pi_1, Cfn, Cfp)
    """

    FNR = conf_matr[0, 1] / (conf_matr[0, 1] + conf_matr[1, 1])
    FPR = conf_matr[1, 0] / (conf_matr[1, 0] + conf_matr[0, 0])

    DCFu = pi_1 * Cfn * FNR + (1 - pi_1) * Cfp * FPR

    Bdummy = min(pi_1 * Cfn, (1 - pi_1) * Cfp)
    DCF = DCFu / Bdummy

    return DCF

def bayes_min_dcf(llr, labels, pi_1, Cfn, Cfp):
    """
    Computes the minimum normalized DCF trying different thresholds in the given range
    :param llr: log-likelihood ratios to classify
    :param labels: labels for building confusion matrix
    :param pi_1: target application prior probability for the true class
    :param Cfn: target application cost for a false negative prediction
    :param Cfp: target application cost for a false positive prediction
    :return: the minimum normalized dcf
    """

    if llr.ndim > 1:
        llr = llr.flatten()

    llr_sorted = np.sort(llr)
    minDCF = np.inf
    best_threshold = 1
    for t in llr_sorted:
        predictions = bayes_binary_optimal_classifier(llr, pi_1, Cfn, Cfp, threshold=t)
        conf_matr = get_confusion_matrix(predictions, labels)
        dcf = bayes_binary_dcf(conf_matr, pi_1, Cfn, Cfp)
        if dcf < minDCF:
            minDCF = dcf
            best_threshold = t

    return minDCF, best_threshold

def draw_ROC(llr, labels):
    """
    Draw the ROC
    :param llr: log-likelihood ratios for comparing against thresholds
    :param labels: labels for building confusion matrix and then FNR/FPR
    :param min_threshold: minimum threshold
    :param max_threshold: maximum threshold
    :param points: number of points in the range of the thresholds to calculate TPR/FPR on
    :return: None
    """
    if llr.ndim > 1:
        llr = llr.flatten()

    llr_sorted = np.sort(llr)
    FPRs = []
    TPRs = []
    for t in llr_sorted:
        predictions = bayes_binary_optimal_classifier(llr, 0.5, 1, 1, threshold=t)
        conf_matr = get_confusion_matrix(predictions, labels)
        FNR = conf_matr[0, 1] / (conf_matr[0, 1] + conf_matr[1, 1])
        TPR = 1 - FNR
        FPR = conf_matr[1, 0] / (conf_matr[1, 0] + conf_matr[0, 0])
        FPRs.append(FPR)
        TPRs.append(TPR)

    plt.title("ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot(FPRs, TPRs)

def draw_NormalizedBayesErrorPlot(llr, labels, p_min, p_max, p_points, recognizer_name=""):
    """
    Draw the Normalized Bayes Error Plots to assess the performance of the classifier as we vary the application
    :param llr: log-likelihood ratios to compare with the different thresholds
    :param labels: ground truths
    :param p_min: minimum of effective prior log odds to calculate dcf and min-dcf on
    :param p_max: maximum of effective prior log odds to calculate dcf and min-dcf on
    :param p_points: points of effective prior log odds to calculate dcf and min-dcf on
    :param t_min: minimum threshold to calculate the dcf min
    :param t_max: maximum threshold to caculate the dcf min
    :param t_points: points of thresholds to calculate the dcf min
    :param recognizer_name: A descriptive name of the recognizer under evaluation, useful for comparing on the same
                            figure different recognizers
    :return: None
    """
    effPriorLogOdds = np.linspace(p_min, p_max, p_points)
    DCFs = []
    minDCFs = []
    for p in effPriorLogOdds:
        effPrior = 1 / (1 + np.exp(-p))
        predictions = bayes_binary_optimal_classifier(llr, effPrior, 1, 1)
        conf_matr = get_confusion_matrix(predictions, labels)
        DCF = bayes_binary_dcf(conf_matr, effPrior, 1, 1)
        minDCF, _ = bayes_min_dcf(llr, labels, effPrior, 1, 1)
        DCFs.append(DCF)
        minDCFs.append(minDCF)

    plt.title("Normalized Bayes Error Plots")
    plt.xlabel("Prior log-odds")
    plt.ylabel("DCF value")
    plt.plot(effPriorLogOdds, DCFs, label="DCF (%s)" % recognizer_name)
    plt.plot(effPriorLogOdds, minDCFs, label="min DCF (%s)" % recognizer_name)
    plt.legend(loc="lower left")
    plt.ylim([0, 1.1])
    plt.xlim([p_min, p_max])

def bayes_multiclass_dcfu(conf_matr, pi, C):
    DCFu = 0.0
    for j in range(conf_matr.shape[0]):
        tmp = 0.0
        for i in range(conf_matr.shape[0]):
            Rij = conf_matr[i, j] / (conf_matr[:, j].sum())
            tmp = tmp + Rij * C[i, j]
        DCFu = DCFu + pi[j] * tmp

    return DCFu

def bayes_multiclass_dcf(conf_matr, pi, C):
    DCFu = bayes_multiclass_dcfu(conf_matr, pi, C)
    Cdummy = np.min(C @ pi)
    DCF = DCFu / Cdummy
    return DCF