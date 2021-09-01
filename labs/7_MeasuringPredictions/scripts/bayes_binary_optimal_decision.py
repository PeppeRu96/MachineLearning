import os
import numpy as np
import matplotlib.pyplot as plt

import evaluation.common as eval

SCRIPT_PATH = os.path.dirname(__file__)
COMMEDIA_INFPAR_LLR_PATH = os.path.join("..", "Data", "commedia_llr_infpar.npy")
COMMEDIA_INFPAR_LABELS_PATH = os.path.join("..", "Data", "commedia_labels_infpar.npy")
COMMEDIA_INFPAR_EPS1_LLR_PATH = os.path.join("..", "Data", "commedia_llr_infpar_eps1.npy")
COMMEDIA_INFPAR_EPS1_LABELS_PATH = os.path.join("..", "Data", "commedia_labels_infpar_eps1.npy")

if __name__ == "__main__":
    llr_infpar = np.load(COMMEDIA_INFPAR_LLR_PATH)
    labels_infpar = np.load(COMMEDIA_INFPAR_LABELS_PATH)

    llr_infpar_eps1 = np.load(COMMEDIA_INFPAR_EPS1_LLR_PATH)
    labels_infpar_eps1 = np.load(COMMEDIA_INFPAR_EPS1_LABELS_PATH)

    # BAYES BINARY OPTIMAL DECISION
    def test_bayes_eval(pi_1, Cfn, Cfp):
        predictions = eval.bayes_binary_optimal_classifier(llr_infpar, pi_1, Cfn, Cfp)
        print("Confusion matrix with pi_1: %f\tCfn: %f\tCfp: %f" % (pi_1, Cfn, Cfp))
        eval.get_confusion_matrix(predictions, labels_infpar, show=True)
        print("")

    confs = [
        (0.5, 1, 1),
        (0.8, 1, 1),
        (0.5, 10, 1),
        (0.8, 1, 10)
    ]

    print("Bayes binary optimal decision evaluation..")
    for (pi_1, Cfn, Cfp) in confs:
        test_bayes_eval(pi_1, Cfn, Cfp)

    # Bayes DCFu
    def test_bayes_dcfu(pi_1, Cfn, Cfp):
        predictions = eval.bayes_binary_optimal_classifier(llr_infpar, pi_1, Cfn, Cfp)
        conf_matr = eval.get_confusion_matrix(predictions, labels_infpar)
        dcfu = eval.bayes_binary_dcfu(conf_matr, pi_1, Cfn, Cfp)
        print("DCFu with an application (pi_1: %.2f, Cfn: %.2f, Cfp: %.2f):\t%.3f" % (pi_1, Cfn, Cfp, dcfu))

    print("Bayes binary DCF unnormalized evaluation..")
    for (pi_1, Cfn, Cfp) in confs:
        test_bayes_dcfu(pi_1, Cfn, Cfp)

    print("")

    # Bayes DCF
    def test_bayes_dcf(pi_1, Cfn, Cfp):
        predictions = eval.bayes_binary_optimal_classifier(llr_infpar, pi_1, Cfn, Cfp)
        conf_matr = eval.get_confusion_matrix(predictions, labels_infpar)
        dcf = eval.bayes_binary_dcf(conf_matr, pi_1, Cfn, Cfp)
        print("DCF with an application (pi_1: %.2f, Cfn: %.2f, Cfp: %.2f):\t%.3f" % (pi_1, Cfn, Cfp, dcf))


    print("Bayes binary DCF evaluation..")
    for (pi_1, Cfn, Cfp) in confs:
        test_bayes_dcf(pi_1, Cfn, Cfp)

    print("")

    # Bayes binary minimum DCF
    def test_bayes_min_dcf(pi_1, Cfn, Cfp):
        minDCF, _ = eval.bayes_min_dcf(llr_infpar, labels_infpar, pi_1, Cfn, Cfp, -20, 20, 1000)

        print("Minimum DCF with an application (pi_1: %.2f, Cfn: %.2f, Cfp: %.2f):\t%.3f" % (pi_1, Cfn, Cfp, minDCF))


    print("Bayes binary minimum DCF evaluation..")
    for (pi_1, Cfn, Cfp) in confs:
        test_bayes_min_dcf(pi_1, Cfn, Cfp)

    print("")

    #plt.figure()
    #eval.draw_ROC(llr_infpar, labels_infpar, -100, 100, 30000)

    #plt.figure()
    #eval.draw_NormalizedBayesErrorPlot(llr_infpar, labels_infpar, -3, 3, 21, -20, 20, 500)

    # Comparing recognizers
    def test_comparing_recognizers(pi_1, Cfn, Cfp):
        # DCF
        predictions = eval.bayes_binary_optimal_classifier(llr_infpar, pi_1, Cfn, Cfp)
        conf_matr = eval.get_confusion_matrix(predictions, labels_infpar)
        dcf = eval.bayes_binary_dcf(conf_matr, pi_1, Cfn, Cfp)

        predictions_eps1 = eval.bayes_binary_optimal_classifier(llr_infpar_eps1, pi_1, Cfn, Cfp)
        conf_matr_eps1 = eval.get_confusion_matrix(predictions_eps1, labels_infpar)
        dcf_eps1 = eval.bayes_binary_dcf(conf_matr_eps1, pi_1, Cfn, Cfp)

        # MIN DCF
        minDCF, _ = eval.bayes_min_dcf(llr_infpar, labels_infpar, pi_1, Cfn, Cfp, -20, 20, 1000)
        minDCF_eps1, _ = eval.bayes_min_dcf(llr_infpar_eps1, labels_infpar_eps1, pi_1, Cfn, Cfp, -20, 20, 1000)

        print("(pi_1: %.1f, Cfn: %.1f, Cfp: %.1f) -\tDCF (eps: 0.001): %.3f\tMin DCF (eps: 0.001): %.3f\tDCF (eps: 1): %.3f\tMin DCF (eps: 1): %.3f" %
              (pi_1, Cfn, Cfp, dcf, minDCF, dcf_eps1, minDCF_eps1))

    print("Comparing two recognizers (eps: 0.001 and eps: 1)..")
    for (pi_1, Cfn, Cfp) in confs:
        test_comparing_recognizers(pi_1, Cfn, Cfp)

    plt.figure()
    eval.draw_NormalizedBayesErrorPlot(llr_infpar, labels_infpar, -3, 3, 21, -20, 20, 500, recognizer_name="eps: 0.001")
    eval.draw_NormalizedBayesErrorPlot(llr_infpar_eps1, labels_infpar_eps1, -3, 3, 21, -20, 20, 500, recognizer_name="eps: 1")

    plt.show()