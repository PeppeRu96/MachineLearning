import matplotlib.pyplot as plt

from wine_project.utility.ds_common import *
import evaluation.common as eval
from classifiers.svm import cross_validate_svm, SVM_Classifier
from classifiers.gmm_classifier import cross_validate_gmm

from classifiers.logistic_regression import LogisticRegressionClassifier

import argparse

TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "svm_gmm")
SVM_GMM_RECALIBRATION_TRAINLOG_FNAME = "svm_gmm_recalibration_trainlog_1.txt"

BAYES_ERROR_PLOT_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm_gmm", "svm_gmm_bayes_error_plot")
BAYES_ERROR_PLOT_UNCALIBRATED_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm_gmm", "svm_gmm_uncalibrated_bayes_error_plot")
BAYES_ERROR_PLOT_CALIBRATED_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm_gmm", "svm_gmm_calibrated_bayes_error_plot")
BAYES_ERROR_PLOT_FUSION_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm_gmm", "svm_gmm_fusion_bayes_error_plot")
ROC_FUSION_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm_gmm", "svm_gmm_fusion_roc")

CROSS_VALIDATION_SVM_SCORES = os.path.join(SCRIPT_PATH, "..", "data", "svm_gmm", "svm_cross_val_scores.npy")
CROSS_VALIDATION_SVM_LABELS = os.path.join(SCRIPT_PATH, "..", "data", "svm_gmm", "svm_cross_val_labels.npy")
CROSS_VALIDATION_GMM_SCORES = os.path.join(SCRIPT_PATH, "..", "data", "svm_gmm", "gmm_cross_val_scores.npy")
CROSS_VALIDATION_GMM_LABELS = os.path.join(SCRIPT_PATH, "..", "data", "svm_gmm", "gmm_cross_val_labels.npy")

create_folder_if_not_exist(os.path.join(TRAINLOGS_BASEPATH, SVM_GMM_RECALIBRATION_TRAINLOG_FNAME))
create_folder_if_not_exist(BAYES_ERROR_PLOT_GRAPH_PATH)
create_folder_if_not_exist(CROSS_VALIDATION_SVM_SCORES)

# Best RBF SVM configuration for our main target application
best_svm_preproc_conf = PreprocessConf([
    PreprocStage(Preproc.Centering),
    PreprocStage(Preproc.Whitening_Within_Covariance),
    PreprocStage(Preproc.L2_Normalization)
])
best_svm_K = 1
best_svm_g = 8
best_svm_C = 0.5
best_svm_train_pi1 = None # class-balancing not active
best_svm_kernel = SVM_Classifier.Kernel_RadialBasisFunction(best_svm_g)

# Best GMM configuration for our main target application
best_gmm_preproc_conf = PreprocessConf([
                PreprocStage(Preproc.Centering),
                PreprocStage(Preproc.Whitening_Within_Covariance),
                PreprocStage(Preproc.L2_Normalization)
            ])
best_gmm_num_components = 256
best_gmm_diag = False
best_gmm_tied = False
# constants
ALPHA = 0.1
PSI = 0.01

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch RBF SVM classificator building",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--bayes_error_plot", type=bool, default=False,
                        help="Display a bayes error plot for the best svm and gmm model")
    parser.add_argument("--lr_recalibration", type=bool, default=False,
                        help="Recalibrate the scores using a linear LR for the best svm and gmm model and fuse them")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    # Cross validation for svm and gmm (best models)
    if os.path.exists(CROSS_VALIDATION_SVM_SCORES) and os.path.exists(CROSS_VALIDATION_SVM_LABELS) and \
        os.path.exists(CROSS_VALIDATION_GMM_SCORES) and os.path.exists(CROSS_VALIDATION_GMM_LABELS):
        print("Loading cross-validation scores and labels from file..")
        svm_scores = np.load(CROSS_VALIDATION_SVM_SCORES)
        svm_labels = np.load(CROSS_VALIDATION_SVM_LABELS)
        gmm_scores = np.load(CROSS_VALIDATION_GMM_SCORES)
        gmm_labels = np.load(CROSS_VALIDATION_GMM_LABELS)
    else:
        print("Cross-validating best SVM and best GMM..")
        svm_scores, svm_labels = cross_validate_svm(folds_data, folds_labels, best_svm_preproc_conf, best_svm_C, best_svm_K,
                                                    specific_pi1=best_svm_train_pi1, kernel=best_svm_kernel)
        gmm_all_scores, gmm_all_labels = cross_validate_gmm(folds_data, folds_labels, best_gmm_preproc_conf, ALPHA, PSI,
                                                            best_gmm_diag,
                                                            best_gmm_tied, best_gmm_num_components, verbose=False)
        gmm_scores, gmm_labels = gmm_all_scores[-1], gmm_all_labels[-1]

        np.save(CROSS_VALIDATION_SVM_SCORES, svm_scores)
        np.save(CROSS_VALIDATION_SVM_LABELS, svm_labels)
        np.save(CROSS_VALIDATION_GMM_SCORES, gmm_scores)
        np.save(CROSS_VALIDATION_GMM_LABELS, gmm_labels)

    # Splitting the scores in calibration set and validation set
    def split_scores(scores, labels):
        """ Split an array of scores (from different classifiers) in a calibration set and validation set
        :param scores: a vector [scores1, scores2, ... scoresN] to split N classifier scores with the same random split
        :return calibration_scores : np.array (#classifiers, #calib_samples)
                calibration_labels : np.array (#calib_samples)
                validation_scores  : np.array (#classifiers, #val_samples)
                validation_labels  : np.array (#val_samples)
        """
        idx = np.random.permutation(labels.shape[0])
        labels_shuffled = labels[idx]
        middle = labels_shuffled.shape[0]//2
        calibration_labels = labels_shuffled[:middle]
        validation_labels = labels_shuffled[middle:]
        print(f"before splitting - #samples: {labels.shape[0]}")
        print(f"before splitting - # Hf: {(labels == 0).sum()}")
        print(f"before splitting - # Ht: {(labels == 1).sum()}")
        print("")
        print("Calibration set:")
        print(f"after splitting - calibration #samples: {calibration_labels.shape[0]}")
        print(f"after splitting - calibration # Hf: {(calibration_labels == 0).sum()}")
        print(f"after splitting - calibration # Ht: {(calibration_labels == 1).sum()}")
        print()
        print("Validation set:")
        print(f"after splitting - validation #samples: {validation_labels.shape[0]}")
        print(f"before splitting - validation # Hf: {(validation_labels == 0).sum()}")
        print(f"before splitting - validation # Ht: {(validation_labels == 1).sum()}")

        calibration_scores = []
        validation_scores = []
        for s in scores:
            if s.ndim < 2:
                s = s.reshape((1, s.shape[0]))
            s_shuffled = s[:, idx]
            calibration_s = s_shuffled[:, :middle]
            validation_s = s_shuffled[:, middle:]
            calibration_scores.append(calibration_s)
            validation_scores.append(validation_s)

        calibration_scores = np.vstack(calibration_scores)
        validation_scores = np.vstack(validation_scores)

        print()
        print(f"Calibration scores shape: {calibration_scores.shape}")
        print(f"Calibration labels shape: {calibration_labels.shape}")
        print(f"Validation scores shape: {validation_scores.shape}")
        print(f"Validation labels shape: {validation_labels.shape}")

        return calibration_scores, calibration_labels, validation_scores, validation_labels

    # Bayes error plot
    if args.bayes_error_plot:
        print("Printing Bayes Error Plot of SVM and GMM..")
        plt.figure(figsize=[13, 9.7])
        eval.draw_NormalizedBayesErrorPlot(svm_scores, svm_labels, -4, 4, 25, recognizer_name="RBF SVM", color="red")
        eval.draw_NormalizedBayesErrorPlot(gmm_scores, gmm_labels, -4, 4, 25, recognizer_name="GMM", color="blue")
        plt.savefig(f"{BAYES_ERROR_PLOT_GRAPH_PATH}")
        print(f"Bayes Error Plot of SVM and GMM saved in {BAYES_ERROR_PLOT_GRAPH_PATH}")

    # Splitting validation scores in calibration set and validation set
    print("Splitting svm and gmm scores in calibration and validation set..............")
    calibration_scores, calibration_labels, validation_scores, validation_labels =\
        split_scores([np.copy(svm_scores), np.copy(gmm_scores)], np.copy(svm_labels))

    print()

    # LR-Recalibration
    def lr_recalibration(calib_scores, calib_labels, val_scores, l, pi1=None):
        lr = LogisticRegressionClassifier(2)
        lr.train(calib_scores, calib_labels, l, pi1=pi1)

        return lr.compute_binary_classifier_llr(val_scores)

    def recover_lr_calibrated_scores(scores, pi1):
        return scores - np.log(pi1 / (1-pi1))

    # ----------------------------------------------------------------------------#

    if args.lr_recalibration:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, SVM_GMM_RECALIBRATION_TRAINLOG_FNAME)):
            # Uncalibrated results
            print("Uncalibrated results on the validation set (validation of the calibration-validation split):")
            for eval_pi1, eval_Cfn, eval_Cfp in applications:
                svm_unc_minDCF, _ = eval.bayes_min_dcf(validation_scores[0:1], validation_labels, eval_pi1, eval_Cfn, eval_Cfp)
                svm_unc_actDCF = eval.bayes_binary_dcf(validation_scores[0:1], validation_labels, eval_pi1, eval_Cfn, eval_Cfp)
                print(f"\tuncalibrated RBF SVM minDCF (π={eval_pi1}): {svm_unc_minDCF:.3f}")
                print(f"\tuncalibrated RBF SVM actDCF (π={eval_pi1}): {svm_unc_actDCF:.3f}")

                gmm_unc_minDCF, _ = eval.bayes_min_dcf(validation_scores[1:2], validation_labels, eval_pi1, eval_Cfn, eval_Cfp)
                gmm_unc_actDCF = eval.bayes_binary_dcf(validation_scores[1:2], validation_labels, eval_pi1, eval_Cfn, eval_Cfp)
                print(f"\tuncalibrated GMM minDCF (π={eval_pi1}): {gmm_unc_minDCF:.3f}")
                print(f"\tuncalibrated GMM actDCF (π={eval_pi1}): {gmm_unc_actDCF:.3f}")
                print()
            print("")

            print("Calibrated results:")
            l = 10 ** (-3)
            svm_lr_scores = lr_recalibration(calibration_scores[0:1], calibration_labels, validation_scores[0:1], l, pi1=applications[0][0])
            gmm_lr_scores = lr_recalibration(calibration_scores[1:2], calibration_labels, validation_scores[1:2], l, pi1=applications[0][0])
            fusion_lr_scores = lr_recalibration(calibration_scores, calibration_labels, validation_scores, l, pi1=applications[0][0])

            # Recover calibrated scores for different target applications
            for train_pi1, train_Cfn, train_Cfp in applications:
                print(f"Recovering calibrated scores for target application π={train_pi1}:")
                svm_calibrated_scores = recover_lr_calibrated_scores(svm_lr_scores, train_pi1)
                gmm_calibrated_scores = recover_lr_calibrated_scores(gmm_lr_scores, train_pi1)
                fusion_calibrated_scores = recover_lr_calibrated_scores(fusion_lr_scores, train_pi1)
                for eval_pi1, eval_Cfn, eval_Cfp in applications:
                    svm_cal_minDCF, _ = eval.bayes_min_dcf(svm_calibrated_scores, validation_labels, eval_pi1, eval_Cfn, eval_Cfp)
                    svm_cal_actDCF = eval.bayes_binary_dcf(svm_calibrated_scores, validation_labels, eval_pi1, eval_Cfn, eval_Cfp)

                    gmm_cal_minDCF, _ = eval.bayes_min_dcf(gmm_calibrated_scores, validation_labels, eval_pi1, eval_Cfn, eval_Cfp)
                    gmm_cal_actDCF = eval.bayes_binary_dcf(gmm_calibrated_scores, validation_labels, eval_pi1, eval_Cfn, eval_Cfp)

                    fusion_minDCF, _ = eval.bayes_min_dcf(fusion_calibrated_scores, validation_labels, eval_pi1, eval_Cfn, eval_Cfp)
                    fusion_actDCF = eval.bayes_binary_dcf(fusion_calibrated_scores, validation_labels, eval_pi1, eval_Cfn, eval_Cfp)

                    print(f"\tA) calibrated RBF SVM minDCF (target π={eval_pi1}): {svm_cal_minDCF:.3f}")
                    print(f"\tA) calibrated RBF SVM actDCF (target π={eval_pi1}): {svm_cal_actDCF:.3f}")
                    print(f"\tB) calibrated GMM minDCF (target π={eval_pi1}): {gmm_cal_minDCF:.3f}")
                    print(f"\tB) calibrated GMM actDCF (target π={eval_pi1}): {gmm_cal_actDCF:.3f}")
                    print(f"\tC) Fusion minDCF (target π={eval_pi1}): {fusion_minDCF:.3f}")
                    print(f"\tC) Fusion actDCF (target π={eval_pi1}): {fusion_actDCF:.3f}")
                    print()
                print()

            # Bayes error plot
            # Without calibration
            print("Printing Bayes Error Plot of SVM and GMM without calibration (on the validation of the calibration-validation split)..")
            plt.figure(figsize=[13, 9.7])
            eval.draw_NormalizedBayesErrorPlot(validation_scores[0:1], validation_labels, -4, 4, 25,
                                               recognizer_name="SVM (uncalibrated)", color="red")
            eval.draw_NormalizedBayesErrorPlot(validation_scores[1:2], validation_labels, -4, 4, 25,
                                               recognizer_name="GMM (uncalibrated)", color="blue")
            plt.savefig(f"{BAYES_ERROR_PLOT_UNCALIBRATED_GRAPH_PATH}")
            print(f"Bayes Error Plot of SVM and GMM saved in {BAYES_ERROR_PLOT_UNCALIBRATED_GRAPH_PATH}")

            # With calibration
            svm_calibrated_scores = recover_lr_calibrated_scores(svm_lr_scores, applications[0][0])
            gmm_calibrated_scores = recover_lr_calibrated_scores(gmm_lr_scores, applications[0][0])
            fusion_calibrated_scores = recover_lr_calibrated_scores(fusion_lr_scores, applications[0][0])
            print("Printing Bayes Error Plot of SVM and GMM after calibration (on the validation of the calibration-validation split)..")

            plt.figure(figsize=[13, 9.7])
            eval.draw_NormalizedBayesErrorPlot(svm_calibrated_scores, validation_labels, -4, 4, 25, recognizer_name="SVM (calibrated)",color="red")
            eval.draw_NormalizedBayesErrorPlot(gmm_calibrated_scores, validation_labels, -4, 4, 25, recognizer_name="GMM (calibrated)", color="blue")
            plt.savefig(f"{BAYES_ERROR_PLOT_CALIBRATED_GRAPH_PATH}")
            print(f"Bayes Error Plot of SVM and GMM saved in {BAYES_ERROR_PLOT_CALIBRATED_GRAPH_PATH}")

            # Bayer Error Plot of Calibrated SVM + Calibrated GMM + Fusion
            print("Printing Bayes Error Plot of calibrated SVM, calibrated GMM and Fusion (on the validation of the calibration-validation split)..")
            plt.figure(figsize=[13, 9.7])
            eval.draw_NormalizedBayesErrorPlot(svm_calibrated_scores, validation_labels, -4, 4, 25,
                                               recognizer_name="SVM (calibrated)", color="red", actDCF_only=True)
            eval.draw_NormalizedBayesErrorPlot(gmm_calibrated_scores, validation_labels, -4, 4, 25,
                                               recognizer_name="GMM (calibrated)", color="blue", actDCF_only=True)
            eval.draw_NormalizedBayesErrorPlot(fusion_calibrated_scores, validation_labels, -4, 4, 25,
                                               recognizer_name="Fusion", color="green")
            plt.savefig(f"{BAYES_ERROR_PLOT_FUSION_GRAPH_PATH}")
            print(f"Bayes Error Plot of SVM and GMM saved in {BAYES_ERROR_PLOT_FUSION_GRAPH_PATH}")

            # ROC of Calibrated SVM + Calibrated GMM + Fusion
            print("Printing ROC of calibrated SVM, calibrated GMM and Fusion (on the validation of the calibration-validation split)..")
            plt.figure(figsize=[13, 9.7])
            eval.draw_ROC(svm_calibrated_scores, validation_labels, color="red", recognizer_name="SVM (calibrated)")
            eval.draw_ROC(gmm_calibrated_scores, validation_labels, color="blue", recognizer_name="GMM (calibrated)")
            eval.draw_ROC(fusion_calibrated_scores, validation_labels, color="green", recognizer_name="Fusion")
            plt.savefig(f"{ROC_FUSION_GRAPH_PATH}")
            print(f"Bayes Error Plot of SVM and GMM saved in {ROC_FUSION_GRAPH_PATH}")

    # ----------------------------------------------------------------------------#

    plt.show()