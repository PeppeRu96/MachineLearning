import matplotlib.pyplot as plt

from wine_project.utility.ds_common import *
import evaluation.common as eval
from classifiers.svm import cross_validate_svm, SVM_Classifier
from classifiers.gmm_classifier import cross_validate_gmm

import argparse

BAYES_ERROR_PLOT_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm_gmm", "svm_gmm_bayes_error_plot")
CROSS_VALIDATION_SVM_SCORES = os.path.join(SCRIPT_PATH, "..", "data", "svm_gmm", "svm_cross_val_scores")
CROSS_VALIDATION_SVM_LABELS = os.path.join(SCRIPT_PATH, "..", "data", "svm_gmm", "svm_cross_val_labels")
CROSS_VALIDATION_GMM_SCORES = os.path.join(SCRIPT_PATH, "..", "data", "svm_gmm", "gmm_cross_val_scores")
CROSS_VALIDATION_GMM_LABELS = os.path.join(SCRIPT_PATH, "..", "data", "svm_gmm", "gmm_cross_val_labels")
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
                        help="Recalibrate the scores using a linear LR for the best svm and gmm model")
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

    # Bayes error plot
    if args.bayes_error_plot:
        print("Printing Bayes Error Plot of SVM and GMM..")
        plt.figure(figsize=[13, 9.7])
        eval.draw_NormalizedBayesErrorPlot(svm_scores, svm_labels, -4, 4, 25, recognizer_name="RBF SVM", color="red")
        eval.draw_NormalizedBayesErrorPlot(gmm_scores, gmm_labels, -4, 4, 25, recognizer_name="GMM", color="blue")
        plt.savefig(f"{BAYES_ERROR_PLOT_GRAPH_PATH}")
        print(f"Bayes Error Plot of SVM and GMM saved in {BAYES_ERROR_PLOT_GRAPH_PATH}")

    # LR-Recalibration
    # Splitting the scores in calibration set and validation set


    if args.lr_recalibration:
        pass
    plt.show()