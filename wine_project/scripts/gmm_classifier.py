import matplotlib.pyplot as plt
import time

import numpy as np

from wine_project.utility.ds_common import *
import evaluation.common as eval

from density_estimation.gaussian_mixture_model import LBG_estimate
from classifiers.gmm_classifier import cross_validate_gmm

import argparse

# TRAIN OUTPUT PATHS
TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "gmm")
GMM_TRAINLOG_FNAME = "gmm_trainlog_1.txt"
GMM_ACTUAL_DCF_TRAINLOG_FNAME = "gmm_actual_dcf_trainlog_1.txt"

GMM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "gmm", "gmm_graph_")

# EVALUATION OUTPUT PATHS
EVAL_TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "gmm", "eval")
# PARTIAL TRAINING DATASET
EVAL_PARTIAL_GMM_TRAINLOG_FNAME = "eval_partial_gmm_trainlog_1.txt"
EVAL_PARTIAL_GMM_ACTUAL_DCF_TRAINLOG_FNAME = "eval_partial_gmm_actual_dcf_trainlog_1.txt"

EVAL_PARTIAL_GMM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "gmm", "eval", "eval_partial_gmm_graph_")

# FULL TRAINING DATASET
EVAL_FULL_GMM_TRAINLOG_FNAME = "eval_full_gmm_trainlog_1.txt"
EVAL_FULL_GMM_ACTUAL_DCF_TRAINLOG_FNAME = "eval_full_gmm_actual_dcf_trainlog_1.txt"

EVAL_FULL_GMM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "gmm", "eval", "eval_full_gmm_graph_")

create_folder_if_not_exist(os.path.join(TRAINLOGS_BASEPATH, "dummy.txt"))
create_folder_if_not_exist(GMM_GRAPH_PATH)
create_folder_if_not_exist(os.path.join(EVAL_TRAINLOGS_BASEPATH, "dummy.txt"))
create_folder_if_not_exist(EVAL_FULL_GMM_GRAPH_PATH)

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch GMM classificator building",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gridsearch", default=False, action="store_true",
                        help="Start a gridsearch cross-validation to optimize with respect to the number of components, the preprocess configuration and the model type")
    parser.add_argument("--actual_dcf", default=False, action="store_true",
                        help="Calculate actual DCF for the different target application using the best model")

    parser.add_argument("--eval_partial_gridsearch", default=False, action="store_true",
                        help="Start a gridsearch cross-validation to optimize with respect to the number of components, the preprocess configuration and the model type")
    parser.add_argument("--eval_partial_actual_dcf", default=False, action="store_true",
                        help="Calculate actual DCF for the different target application using the best model")

    parser.add_argument("--eval_full_gridsearch", default=False, action="store_true",
                        help="Start a gridsearch cross-validation to optimize with respect to the number of components, the preprocess configuration and the model type")
    parser.add_argument("--eval_full_actual_dcf", default=False, action="store_true",
                        help="Calculate actual DCF for the different target application using the best model")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    # Load the test dataset
    X_test, y_test = load_dataset(train=False, only_data=True)

    # constants
    ALPHA = 0.1
    PSI = 0.01

    def gmm_gridsearch(X_train, y_train, X_test=None, y_test=None, partial=False):
        # Preprocessing configurations to try
        preproc_configurations = [
            PreprocessConf([]),
            PreprocessConf([PreprocStage(Preproc.Gaussianization)]),
            PreprocessConf([
                PreprocStage(Preproc.Centering),
                PreprocStage(Preproc.Whitening_Covariance),
                PreprocStage(Preproc.L2_Normalization)
            ]),
            PreprocessConf([
                PreprocStage(Preproc.Centering),
                PreprocStage(Preproc.Whitening_Within_Covariance),
                PreprocStage(Preproc.L2_Normalization)
            ]),
        ]
        # preproc_configurations = [
        #     PreprocessConf([])
        # ]

        # Grid hyperparameters
        diags = [False, True]
        tieds = [False, True]
        max_comps = 256
        logcomps = int(np.log2(max_comps))
        comps = np.logspace(0, logcomps, logcomps+1, base=2)

        minDCFs = np.zeros((len(preproc_configurations), len(diags), len(tieds), len(comps), len(applications)))

        # Grid search
        tot_time_start = time.perf_counter()
        print("Grid search on GMM started.")
        #tot_iterations_required = len(preproc_configurations) * len(diags) * len(tieds) * len(comps)
        tot_gs_iterations_required = len(preproc_configurations) * len(diags) * len(tieds)
        #print("Total GMM components trained required ", tot_iterations_required)
        print("Total grid search iterations required ", tot_gs_iterations_required)
        grid_search_iterations = 1
        for conf_i, conf in enumerate(preproc_configurations):
            for d_i, diag in enumerate(diags):
                for t_i, tied in enumerate(tieds):
                    print("Grid search iteration %d / %d" % (grid_search_iterations, tot_gs_iterations_required))
                    time_start = time.perf_counter()
                    scores, labels, _, _ = cross_validate_gmm(conf, ALPHA, PSI, diag, tied, max_comps, X_train=X_train,
                                                        y_train=y_train, X_test=X_test, y_test=y_test,
                                                        X_val=None, y_val=None, verbose=True)
                    for ci, c in enumerate(comps):
                        for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                            minDCF, _ = eval.bayes_min_dcf(scores[ci], labels[ci], pi1, Cfn, Cfp)
                            print("\t\tGMM with %d components min DCF (π=%.1f) : %.3f" % (c, pi1, minDCF))
                            minDCFs[conf_i, d_i, t_i, ci, app_i] = minDCF
                        print("")
                    time_end = time.perf_counter()
                    grid_search_iterations += 1
                    print("Grid search iteration ended in %d seconds" % (time_end - time_start))
        tot_time_end = time.perf_counter()
        print("Grid search on GMM without ended in %d seconds" % (tot_time_end - tot_time_start))
        # Plot
        for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
            for d_i, diag in enumerate(diags):
                for t_i, tied in enumerate(tieds):
                    plt.figure(figsize=[13, 9.7])
                    diag_str = "Diag" if diag else ""
                    tied_str = "Tied" if tied else ""
                    title = "{} {} GMM (π={:.1f})".format(diag_str, tied_str, pi1)
                    plt.title(title)
                    plt.xlabel("Components")
                    plt.ylabel("minDCF")
                    plt.xscale('log', base=2)
                    for conf_i, conf in enumerate(preproc_configurations):
                        y = minDCFs[conf_i, d_i, t_i, :, app_i]
                        width = [0.1 * c for c in comps]
                        x = [c + (conf_i * 0.2 * c) for c in comps]
                        label = conf.to_compact_string() if conf.to_compact_string() != "" else "no-preproc"
                        plt.bar(x, y, label=label, width=width)
                    plt.legend()
                    pi1_without_points = ("%.1f" % pi1).replace(".", "")
                    pi1_str = "pi1-%s" % pi1_without_points
                    diag_str = "_diag" if diag else ""
                    tied_str = "_tied" if tied else ""
                    if X_test is not None:
                        if partial:
                            base_path = EVAL_PARTIAL_GMM_GRAPH_PATH
                        else:
                            base_path = EVAL_FULL_GMM_GRAPH_PATH
                    else:
                        base_path = GMM_GRAPH_PATH

                    full_path = "%s%s%s%s" % (base_path, pi1_str, diag_str, tied_str)

                    plt.savefig(full_path)
                    print(f"Plot saved in {full_path}.")

    # ------------------------------------------------------------------------------------- #
    # TRAIN
    if args.gridsearch:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, GMM_TRAINLOG_FNAME)):
            print("Grid search Cross-Validation on the training dataset for GMM")
            gmm_gridsearch(X_train=folds_data, y_train=folds_labels, X_test=None, y_test=None, partial=False)
    # EVALUATION
    if args.eval_partial_gridsearch:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_GMM_TRAINLOG_FNAME)):
            print("Grid search training on partial train dataset and evaluating on eval dataset for GMM")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            gmm_gridsearch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, partial=True)
    if args.eval_full_gridsearch:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_GMM_TRAINLOG_FNAME)):
            print("Grid search training on the full train dataset and evaluating on eval dataset for GMM")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            gmm_gridsearch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, partial=False)

    # ------------------------------------------------------------------------------------- #

    best_preproc_conf = PreprocessConf([
                PreprocStage(Preproc.Centering),
                PreprocStage(Preproc.Whitening_Within_Covariance),
                PreprocStage(Preproc.L2_Normalization)
            ])
    best_num_components = 256
    best_diag = False
    best_tied = False

    # TRAIN
    if args.actual_dcf:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, GMM_ACTUAL_DCF_TRAINLOG_FNAME)):
            print("Actual DCF for the different target application calculated after cross-validating on the training dataset")
            scores, labels, _, _ = cross_validate_gmm(best_preproc_conf, ALPHA, PSI, best_diag, best_tied, best_num_components,
                                                X_train=folds_data, y_train=folds_labels, X_test=None, y_test=None,
                                                      X_val=None, y_val=None, verbose=True)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores[-1], labels[-1], pi1, Cfn, Cfp)
                actDCF = eval.bayes_binary_dcf(scores[-1], labels[-1], pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                print("\t\tact DCF (π=%.1f) : %.3f" % (pi1, actDCF))
                print()

    # EVAL
    if args.eval_partial_actual_dcf:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_GMM_ACTUAL_DCF_TRAINLOG_FNAME)):
            print("Actual DCF for the different target application calculated training on a partial train dataset and validating on the eval dataset")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            scores, labels, _, _ = cross_validate_gmm(best_preproc_conf, ALPHA, PSI, best_diag, best_tied, best_num_components,
                                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                X_val=None, y_val=None, verbose=True)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores[-1], labels[-1], pi1, Cfn, Cfp)
                actDCF = eval.bayes_binary_dcf(scores[-1], labels[-1], pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                print("\t\tact DCF (π=%.1f) : %.3f" % (pi1, actDCF))
                print()

    if args.eval_full_actual_dcf:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_GMM_ACTUAL_DCF_TRAINLOG_FNAME)):
            print("Actual DCF for the different target application calculated training on a the full train dataset and validating on the eval dataset")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            scores, labels, _, _ = cross_validate_gmm(best_preproc_conf, ALPHA, PSI, best_diag, best_tied, best_num_components,
                                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                      X_val=None, y_val=None, verbose=True)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores[-1], labels[-1], pi1, Cfn, Cfp)
                actDCF = eval.bayes_binary_dcf(scores[-1], labels[-1], pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                print("\t\tact DCF (π=%.1f) : %.3f" % (pi1, actDCF))
                print()

    plt.show()
