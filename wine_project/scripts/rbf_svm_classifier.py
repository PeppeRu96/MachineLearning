import matplotlib.pyplot as plt
import time
from wine_project.utility.ds_common import *
import evaluation.common as eval

import numpy as np


from classifiers.svm import cross_validate_svm, SVM_Classifier

import argparse

# TRAIN OUTPUT PATHS
TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "svm", "rbf")
RBF_SVM_TRAINLOG_FNAME = "rbf_svm_trainlog_1.txt"
RBF_SVM_FINE_GRAINED_TRAINLOG_FNAME = "rbf_svm_fine_grained_trainlog_1.txt"
RBF_SVM_CLASS_BALANCING_PI05_TRAINLOG_FNAME = "rbf_svm_class_balancing_pi05_trainlog_1.txt"
RBF_SVM_CLASS_BALANCING_PI01_TRAINLOG_FNAME = "rbf_svm_class_balancing_pi01_trainlog_1.txt"
RBF_SVM_CLASS_BALANCING_PI09_TRAINLOG_FNAME = "rbf_svm_class_balancing_pi09_trainlog_1.txt"
RBF_SVM_ACTUAL_DCF_TRAINLOG_FNAME = "rbf_svm_actual_dcf_trainlog_1.txt"

RBF_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "rbf", "rbf_svm_graph_")

#EVALUATION OUTPUT PATHS
EVAL_TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "svm", "rbf", "eval")
# PARTIAL TRAINING DATASET
EVAL_PARTIAL_RBF_SVM_TRAINLOG_FNAME = "eval_partial_rbf_svm_trainlog_1.txt"
EVAL_PARTIAL_RBF_SVM_FINE_GRAINED_TRAINLOG_FNAME = "eval_partial_rbf_svm_fine_grained_trainlog_1.txt"
EVAL_PARTIAL_RBF_SVM_CLASS_BALANCING_PI05_TRAINLOG_FNAME = "eval_partial_rbf_svm_class_balancing_pi05_trainlog_1.txt"
EVAL_PARTIAL_RBF_SVM_CLASS_BALANCING_PI01_TRAINLOG_FNAME = "eval_partial_rbf_svm_class_balancing_pi01_trainlog_1.txt"
EVAL_PARTIAL_RBF_SVM_CLASS_BALANCING_PI09_TRAINLOG_FNAME = "eval_partial_rbf_svm_class_balancing_pi09_trainlog_1.txt"
EVAL_PARTIAL_RBF_SVM_ACTUAL_DCF_TRAINLOG_FNAME = "eval_partial_rbf_svm_actual_dcf_trainlog_1.txt"

EVAL_PARTIAL_RBF_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "rbf", "eval", "eval_partial_rbf_svm_graph_")

# FULL TRAINING DATASET
EVAL_FULL_RBF_SVM_TRAINLOG_FNAME = "eval_full_rbf_svm_trainlog_1.txt"
EVAL_FULL_RBF_SVM_FINE_GRAINED_TRAINLOG_FNAME = "eval_full_rbf_svm_fine_grained_trainlog_1.txt"
EVAL_FULL_RBF_SVM_CLASS_BALANCING_PI05_TRAINLOG_FNAME = "eval_full_rbf_svm_class_balancing_pi05_trainlog_1.txt"
EVAL_FULL_RBF_SVM_CLASS_BALANCING_PI01_TRAINLOG_FNAME = "eval_full_rbf_svm_class_balancing_pi01_trainlog_1.txt"
EVAL_FULL_RBF_SVM_CLASS_BALANCING_PI09_TRAINLOG_FNAME = "eval_full_rbf_svm_class_balancing_pi09_trainlog_1.txt"
EVAL_FULL_RBF_SVM_ACTUAL_DCF_TRAINLOG_FNAME = "eval_full_rbf_svm_actual_dcf_trainlog_1.txt"

EVAL_FULL_RBF_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "rbf", "eval", "eval_full_rbf_svm_graph_")

create_folder_if_not_exist(TRAINLOGS_BASEPATH)
create_folder_if_not_exist(os.path.join(TRAINLOGS_BASEPATH, "dummy.txt"))

create_folder_if_not_exist(os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "rbf"))
create_folder_if_not_exist(RBF_SVM_GRAPH_PATH)

create_folder_if_not_exist(os.path.join(EVAL_TRAINLOGS_BASEPATH, "dummy.txt"))
create_folder_if_not_exist(EVAL_FULL_RBF_SVM_GRAPH_PATH)

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch RBF SVM classificator building",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gridsearch", type=bool, default=False,
                        help="Start a coarse-level gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--gridsearch_fine_grained", type=bool, default=False,
                        help="Start a fine-grained gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--class_balancing", type=bool, default=False,
                        help="Start cross-validation to try class-balancing with the best hyperparameters")
    parser.add_argument("--actual_dcf", type=bool, default=False,
                        help="Calculate actual DCF for the different target application using the best model")

    parser.add_argument("--eval_partial_gridsearch", type=bool, default=False,
                        help="Start a coarse-level gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--eval_partial_gridsearch_fine_grained", type=bool, default=False,
                        help="Start a fine-grained gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--eval_partial_class_balancing", type=bool, default=False,
                        help="Start cross-validation to try class-balancing with the best hyperparameters")
    parser.add_argument("--eval_partial_actual_dcf", type=bool, default=False,
                        help="Calculate actual DCF for the different target application using the best model")

    parser.add_argument("--eval_full_gridsearch", type=bool, default=False,
                        help="Start a coarse-level gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--eval_full_gridsearch_fine_grained", type=bool, default=False,
                        help="Start a fine-grained gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--eval_full_class_balancing", type=bool, default=False,
                        help="Start cross-validation to try class-balancing with the best hyperparameters")
    parser.add_argument("--eval_full_actual_dcf", type=bool, default=False,
                        help="Calculate actual DCF for the different target application using the best model")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    # Load the test dataset
    X_test, y_test = load_dataset(train=False, only_data=True)

    def plot_against_C_gamma(conf, K, Cs, gs, X_train, y_train, X_test=None, y_test=None, partial=False, specific_pi1=None, prefix_title=""):

        pi1_str = "with prior weight specific training (π=%.1f)" % (specific_pi1) if specific_pi1 is not None else ""
        minDCFs = np.zeros((len(Cs), len(gs), len(applications)))
        for Ci, C in enumerate(Cs):
            for gi, g in enumerate(gs):
                kernel = SVM_Classifier.Kernel_RadialBasisFunction(g)
                if X_test is None:
                    print("\t(Ci: {}) - 5-Fold Cross-Validation RBF SVM (gamma={:.0e}) {} (C={:.0e} - K={:.1f}) - Preprocessing: {}".format(
                        Ci, g, pi1_str, C, K, conf))
                else:
                    print("\t(Ci: {}) - Train and validation (eval) RBF SVM (gamma={:.0e}) {} (C={:.0e} - K={:.1f}) - Preprocessing: {}".format(
                            Ci, g, pi1_str, C, K, conf))
                time_start = time.perf_counter()
                scores, labels, _, _ = cross_validate_svm(conf, C, K, X_train=X_train, y_train=y_train, X_test=X_test,
                                                    y_test=y_test, X_val=None, y_val=None, specific_pi1=specific_pi1, kernel=kernel)
                for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                    minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                    print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                    minDCFs[Ci, gi, app_i] = minDCF
                time_end = time.perf_counter()
                print("\t\ttime passed: %d seconds" % (time_end - time_start))

        # Create a plot for each target application
        for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
            plt.figure(figsize=[13, 9.7])
            pi1_str = " - train-π: %.1f" % specific_pi1 if specific_pi1 is not None else ""
            title = "{} RBF SVM (K: {:.1f}{}) - {} - target-π={:.1f}".format(prefix_title, K, pi1_str, conf.to_compact_string(), pi1)
            plt.title(title)
            plt.xlabel("C")
            plt.ylabel("minDCF")
            plt.xscale('log')
            x = Cs

            # Plot only some important values of gamma
            if len(gs) > 3:
                # Plot the three top gamma results
                def f_cmp(tup):
                    g_i = tup[0]
                    ys = minDCFs[:, g_i, app_i].flatten()
                    y_min = min(ys)
                    return y_min

                g_sorted = sorted(enumerate(gs), key=f_cmp)
                gs_to_plot = g_sorted[:3]
            else:
                gs_to_plot = [(gi, g) for gi, g in enumerate(gs)]

            for gi, g in gs_to_plot:
                y = minDCFs[:, gi, app_i].flatten()
                gamma_str = f"{int(np.log10(g))}" if float(g).is_integer() else f"{np.log10(g):.1f}"
                plt.plot(x, y, label=f"log(γ)={gamma_str}")
            plt.legend()

            if specific_pi1 is not None:
                pi1_without_points = "%.1f" % specific_pi1
                pi1_without_points = pi1_without_points.replace(".", "")
            pi1_str = "_train-pi1-%s" % pi1_without_points if specific_pi1 is not None else ""

            pi1_target_without_points = f"{pi1:.1f}"
            pi1_target_without_points = pi1_target_without_points.replace(".", "")
            target_pi1_str = f"_target-pi1-{pi1_target_without_points}"

            Kstr = "%.1f" % K
            Kstr = Kstr.replace(".", "-")
            Kstr = "K-" + Kstr

            if X_test is not None:
                if partial:
                    base_path = EVAL_PARTIAL_RBF_SVM_GRAPH_PATH
                else:
                    base_path = EVAL_FULL_RBF_SVM_GRAPH_PATH
            else:
                base_path = RBF_SVM_GRAPH_PATH

            full_path = "%s%s_%s%s%s_%s" % (base_path, Kstr, conf.to_compact_string(), pi1_str, target_pi1_str, prefix_title)
            plt.savefig(full_path)
            print(f"Plot saved in {full_path}.")

    def rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, X_train, y_train, X_test=None, y_test=None, partial=False, prefix_title=""):
        # Grid search without class-balacing
        tot_time_start = time.perf_counter()
        print("Grid search on RBF SVM without class balancing started.")
        tot_iterations_required = len(preproc_configurations) * len(Ks) * len(gamma) * len(Cs)
        tot_gs_iterations_required = len(preproc_configurations) * len(Ks)
        print("Total RBF SVM cross-validation required ", tot_iterations_required)
        print("Total grid search iterations required ", tot_gs_iterations_required)
        grid_search_iterations = 1
        for conf_i, conf in enumerate(preproc_configurations):
                for Ki, K in enumerate(Ks):
                    print("Grid search iteration %d / %d" % (grid_search_iterations, tot_gs_iterations_required))
                    time_start = time.perf_counter()
                    plot_against_C_gamma(conf, K, Cs, gamma, X_train=X_train, y_train=y_train, X_test=X_test,
                                         y_test=y_test, partial=partial, specific_pi1=None, prefix_title=prefix_title)
                    time_end = time.perf_counter()
                    grid_search_iterations += 1
                    print("Grid search iteration ended in %d seconds" % (time_end - time_start))
        tot_time_end = time.perf_counter()
        print("Grid search on RBF SVM without class balancing ended in %d seconds" % (tot_time_end - tot_time_start))

    def rbf_svm_class_balancing(preproc_conf, K, g, C, X_train, y_train, X_test=None, y_test=None):
        kernel = SVM_Classifier.Kernel_RadialBasisFunction(g)

        # Then, we try the best hyperparameters but now class-balancing with respect to the target application
        print("Trying the best hyperparameters but class-balancing w.r.t target applications..")
        for app_i, (train_pi1, Cfn, Cfp) in enumerate(applications):
            if X_test is None:
                print(f"RBF SVM cross-validation with class-balancing for the target application with π={train_pi1:.1f} (gamma={g:.0e}) (C={C:.0e} - K={K:.1f}) - Preprocessing: {preproc_conf}")
            else:
                print(f"RBF SVM training on the train dataset and evaluating on the eval dataset with class-balancing for the target application with π={train_pi1:.1f} (gamma={g:.0e}) (C={C:.0e} - K={K:.1f}) - Preprocessing: {preproc_conf}")
            time_start = time.perf_counter()
            scores, labels, _, _ = cross_validate_svm(preproc_conf, C, K, X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, X_val=None, y_val=None, specific_pi1=train_pi1, kernel=kernel)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
            time_end = time.perf_counter()
            print("Target application (π=%.1f) specific training cross-validation ended in %d seconds" % (
            train_pi1, (time_end - time_start)))
        print("Operation finished")

    # TRAIN
    # Coarse-level grid search to select the best hyperparameters
    if args.gridsearch or args.eval_partial_gridsearch or args.eval_full_gridsearch:
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
        # Coarse-level grid rbf svm hyperparameters
        Ks = [1]
        Cs = np.logspace(-3, 5, 9)
        gamma = np.logspace(-3, 3, 7)
        if args.gridsearch:
            with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_TRAINLOG_FNAME)):
                print("Coarse-level grid search cross-validation on the training dataset for RBF SVM")
                rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, X_train=folds_data, y_train=folds_labels,
                                   X_test=None, y_test=None, partial=False, prefix_title="")
        if args.eval_partial_gridsearch:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_RBF_SVM_TRAINLOG_FNAME)):
                print("Coarse-level grid search training on partial train dataset and evaluating on the eval dataset for RBF SVM")
                X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
                rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test, partial=True, prefix_title="")

        if args.eval_full_gridsearch:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_RBF_SVM_TRAINLOG_FNAME)):
                print("Coarse-level grid search training on the full train dataset and evaluating on the eval dataset for RBF SVM")
                X_train, y_train = concat_kfolds(folds_data, folds_labels)
                rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test, partial=False, prefix_title="")

    # ----------------------------------------------------------------------- #

    # Fine-grained grid search to select the best hyperparameters
    if args.gridsearch_fine_grained or args.eval_partial_gridsearch_fine_grained or args.eval_full_gridsearch_fine_grained:
        # Preprocessing configurations to try
        preproc_configurations = [
            PreprocessConf([
                PreprocStage(Preproc.Centering),
                PreprocStage(Preproc.Whitening_Within_Covariance),
                PreprocStage(Preproc.L2_Normalization)
            ])
        ]

        # Fine-grained grid rbf svm hyperparameters
        Ks = [1]
        Cs = np.logspace(-1, 1, 10)
        gamma = np.logspace(0, 2, 10)

        if args.gridsearch_fine_grained:
            with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_FINE_GRAINED_TRAINLOG_FNAME)):
                print("Fine-grained grid search cross-validation on the training dataset for RBF SVM")
                rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, X_train=folds_data, y_train=folds_labels,
                                   X_test=None, y_test=None, partial=False, prefix_title="fine-grained")
        if args.eval_partial_gridsearch_fine_grained:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_RBF_SVM_FINE_GRAINED_TRAINLOG_FNAME)):
                print("Fine-grained grid search training on partial train dataset and evaluating on the eval dataset for RBF SVM")
                X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
                rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test, partial=True, prefix_title="fine-grained")
        if args.eval_full_gridsearch_fine_grained:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_RBF_SVM_FINE_GRAINED_TRAINLOG_FNAME)):
                print("Fine-grained grid search training on the full train dataset and evaluating on the eval dataset for RBF SVM")
                X_train, y_train = concat_kfolds(folds_data, folds_labels)
                rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test, partial=False, prefix_title="fine-grained")
    # -------------------------------------------------------------------------- #

    # Best configuration for our main target application
    best_preproc_conf = PreprocessConf([
        PreprocStage(Preproc.Centering),
        PreprocStage(Preproc.Whitening_Within_Covariance),
        PreprocStage(Preproc.L2_Normalization)
    ])
    best_K = 1
    best_g = 8
    best_C = 0.5

    if args.class_balancing or args.eval_partial_class_balancing or args.eval_full_class_balancing:
        # pi05 best model - select the best preproc configuration, gamma and C value
        preproc_conf = best_preproc_conf
        K = best_K
        g = best_g
        C = best_C
        if args.class_balancing:
            with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_CLASS_BALANCING_PI05_TRAINLOG_FNAME)):
                print("Cross-Validation on the training dataset for RBF SVM class-balancing with a prior (best model for the first target application)")
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=folds_data, y_train=folds_labels, X_test=None, y_test=None)
        if args.eval_partial_class_balancing:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_RBF_SVM_CLASS_BALANCING_PI05_TRAINLOG_FNAME)):
                print("Training on partial train dataset and evaluating on the eval dataset for RBF SVM class-balancing with a prior (best model for the first target application)")
                X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        if args.eval_full_class_balancing:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_RBF_SVM_CLASS_BALANCING_PI05_TRAINLOG_FNAME)):
                print("Training on the full train dataset and evaluating on the eval dataset for RBF SVM class-balancing with a prior (best model for the first target application)")
                X_train, y_train = concat_kfolds(folds_data, folds_labels)
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # pi01 best model - select the best preproc configuration, gamma and C value
        preproc_conf = PreprocessConf([
            PreprocStage(Preproc.Centering),
            PreprocStage(Preproc.Whitening_Within_Covariance),
            PreprocStage(Preproc.L2_Normalization)
        ])
        K = 1
        g = 10
        C = 0.1
        if args.class_balancing:
            with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_CLASS_BALANCING_PI01_TRAINLOG_FNAME)):
                print(
                    "Cross-Validation on the training dataset for RBF SVM class-balancing with a prior (best model for the second target application)")
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=folds_data, y_train=folds_labels, X_test=None,
                                        y_test=None)
        if args.eval_partial_class_balancing:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH,
                                                 EVAL_PARTIAL_RBF_SVM_CLASS_BALANCING_PI01_TRAINLOG_FNAME)):
                print(
                    "Training on partial train dataset and evaluating on the eval dataset for RBF SVM class-balancing with a prior (best model for the second target application)")
                X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=X_train, y_train=y_train, X_test=X_test,
                                        y_test=y_test)
        if args.eval_full_class_balancing:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH,
                                                 EVAL_FULL_RBF_SVM_CLASS_BALANCING_PI01_TRAINLOG_FNAME)):
                print(
                    "Training on the full train dataset and evaluating on the eval dataset for RBF SVM class-balancing with a prior (best model for the second target application)")
                X_train, y_train = concat_kfolds(folds_data, folds_labels)
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=X_train, y_train=y_train, X_test=X_test,
                                        y_test=y_test)

        # pi09 best model - select the best preproc configuration, gamma and C value
        preproc_conf = PreprocessConf([
            PreprocStage(Preproc.Centering),
            PreprocStage(Preproc.Whitening_Within_Covariance),
            PreprocStage(Preproc.L2_Normalization)
        ])
        K = 1
        g = 10
        C = 1
        if args.class_balancing:
            with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_CLASS_BALANCING_PI09_TRAINLOG_FNAME)):
                print(
                    "Cross-Validation on the training dataset for RBF SVM class-balancing with a prior (best model for the third target application)")
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=folds_data, y_train=folds_labels, X_test=None,
                                        y_test=None)
        if args.eval_partial_class_balancing:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH,
                                                 EVAL_PARTIAL_RBF_SVM_CLASS_BALANCING_PI09_TRAINLOG_FNAME)):
                print(
                    "Training on partial train dataset and evaluating on the eval dataset for RBF SVM class-balancing with a prior (best model for the third target application)")
                X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=X_train, y_train=y_train, X_test=X_test,
                                        y_test=y_test)
        if args.eval_full_class_balancing:
            with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH,
                                                 EVAL_FULL_RBF_SVM_CLASS_BALANCING_PI09_TRAINLOG_FNAME)):
                print(
                    "Training on the full train dataset and evaluating on the eval dataset for RBF SVM class-balancing with a prior (best model for the third target application)")
                X_train, y_train = concat_kfolds(folds_data, folds_labels)
                rbf_svm_class_balancing(preproc_conf, K, g, C, X_train=X_train, y_train=y_train, X_test=X_test,
                                        y_test=y_test)

    # -------------------------------------------------------------------------- #

    # Calculate actual DCF for the different target applications using the best model
    kernel = SVM_Classifier.Kernel_RadialBasisFunction(best_g)
    if args.actual_dcf:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_ACTUAL_DCF_TRAINLOG_FNAME)):
            print("Actual DCF for the different target application calculated after cross-validating on the training dataset")
            scores, labels, _, _ = cross_validate_svm(best_preproc_conf, best_C, best_K, X_train=folds_data, y_train=folds_labels, X_test=None,
                                                y_test=None, X_val=None, y_val=None, specific_pi1=None, kernel=kernel)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                actDCF = eval.bayes_binary_dcf(scores, labels, pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                print("\t\tact DCF (π=%.1f) : %.3f" % (pi1, actDCF))
                print()

    if args.eval_partial_actual_dcf:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_RBF_SVM_ACTUAL_DCF_TRAINLOG_FNAME)):
            print("Actual DCF for the different target application calculated training on a partial train dataset and validating on the eval dataset")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            scores, labels, _, _ = cross_validate_svm(best_preproc_conf, best_C, best_K, X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, X_val=None, y_val=None, specific_pi1=None, kernel=kernel)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                actDCF = eval.bayes_binary_dcf(scores, labels, pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                print("\t\tact DCF (π=%.1f) : %.3f" % (pi1, actDCF))
                print()

    if args.eval_full_actual_dcf:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_RBF_SVM_ACTUAL_DCF_TRAINLOG_FNAME)):
            print("Actual DCF for the different target application calculated training on a the full train dataset and validating on the eval dataset")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            scores, labels, _, _ = cross_validate_svm(best_preproc_conf, best_C, best_K, X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, X_val=None, y_val=None, specific_pi1=None, kernel=kernel)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                actDCF = eval.bayes_binary_dcf(scores, labels, pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                print("\t\tact DCF (π=%.1f) : %.3f" % (pi1, actDCF))
                print()
    # -------------------------------------------------------------------------- #

    plt.show()
