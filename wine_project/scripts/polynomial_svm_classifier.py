import matplotlib.pyplot as plt
import time

from wine_project.utility.ds_common import *
import evaluation.common as eval

from classifiers.svm import cross_validate_svm, SVM_Classifier

import argparse

# TRAIN OUTPUT PATHS
TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "svm", "polynomial")
POLYNOMIAL_SVM_TRAINLOG_FNAME = "polynomial_svm_trainlog_1.txt"
POLYNOMIAL_SVM_CLASS_BALANCING_TRAINLOG_FNAME = "polynomial_svm_class_balancing_trainlog_1.txt"

POLYNOMIAL_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "polynomial", "polynomial_svm_graph_")

# EVALUATION OUTPUT PATHS
EVAL_TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "svm", "polynomial", "eval")
# PARTIAL TRAINING DATASET
EVAL_PARTIAL_POLYNOMIAL_SVM_TRAINLOG_FNAME = "eval_partial_polynomial_svm_trainlog_1.txt"
EVAL_PARTIAL_POLYNOMIAL_SVM_CLASS_BALANCING_TRAINLOG_FNAME = "eval_partial_polynomial_svm_class_balancing_trainlog_1.txt"

EVAL_PARTIAL_POLYNOMIAL_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "polynomial", "eval", "eval_partial_polynomial_svm_graph_")
# PARTIAL TRAINING DATASET
EVAL_FULL_POLYNOMIAL_SVM_TRAINLOG_FNAME = "eval_full_polynomial_svm_trainlog_1.txt"
EVAL_FULL_POLYNOMIAL_SVM_CLASS_BALANCING_TRAINLOG_FNAME = "eval_full_polynomial_svm_class_balancing_trainlog_1.txt"

EVAL_FULL_POLYNOMIAL_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "polynomial", "eval", "eval_full_polynomial_svm_graph_")

create_folder_if_not_exist(TRAINLOGS_BASEPATH)
create_folder_if_not_exist(os.path.join(TRAINLOGS_BASEPATH, "dummy.txt"))
create_folder_if_not_exist(os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "polynomial"))
create_folder_if_not_exist(POLYNOMIAL_SVM_GRAPH_PATH)

create_folder_if_not_exist(os.path.join(EVAL_TRAINLOGS_BASEPATH, "dummy.txt"))
create_folder_if_not_exist(EVAL_PARTIAL_POLYNOMIAL_SVM_GRAPH_PATH)


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch Logistic Regression classificator building",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gridsearch", type=bool, default=False,
                        help="Start a cross-validation grid search to select the best preprocess configuration and the best C")
    parser.add_argument("--class_balancing", type=bool, default=False,
                        help="Start a cross-validation for a polynomial svm model rebalancing the classes to embed a specific prior")

    parser.add_argument("--eval_partial_gridsearch", type=bool, default=False,
                        help="Start a gridsearch, training on partial training dataset and validating on the evaluation dataset to select the best preprocess configuration and the best C")
    parser.add_argument("--eval_partial_class_balancing", type=bool, default=False,
                        help="Start a training on partial training dataset and validating on the evaluation dataset for a linear svm model rebalancing the classes to embed a specific prior")

    parser.add_argument("--eval_full_gridsearch", type=bool, default=False,
                        help="Start a gridsearch, training on the full training dataset and validating on the evaluation dataset to select the best preprocess configuration and the best C")
    parser.add_argument("--eval_full_class_balancing", type=bool, default=False,
                        help="Start a training on the full training dataset and validating on the evaluation dataset for a linear svm model rebalancing the classes to embed a specific prior")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    # Load the test dataset
    X_test, y_test = load_dataset(train=False, only_data=True)

    def plot_against_C(conf, K, Cs, d, c, X_train, y_train, X_test=None, y_test=None, partial=False, specific_pi1=None):
        kernel = SVM_Classifier.Kernel_Polynomial(d, c)

        pi1_str = "with prior weight specific training (π=%.1f)" % (specific_pi1) if specific_pi1 is not None else ""
        minDCFs = np.zeros((len(Cs), len(applications)))
        for Ci, C in enumerate(Cs):
            if X_test is None:
                print("\t(Ci: {}) - 5-Fold Cross-Validation Polynomial SVM (d={} - c={}) {} (C={:.0e} - K={:.1f}) - Preprocessing: {}".format(
                    Ci, d, c, pi1_str, C, K, conf))
            else:
                print("\t(Ci: {}) - Train and validation (eval) Polynomial SVM (d={} - c={}) {} (C={:.0e} - K={:.1f}) - Preprocessing: {}".format(
                        Ci, d, c, pi1_str, C, K, conf))
            time_start = time.perf_counter()
            scores, labels, _, _ = cross_validate_svm(conf, C, K, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                X_val=None, y_val=None, specific_pi1=specific_pi1, kernel=kernel)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                minDCFs[Ci, app_i] = minDCF
            time_end = time.perf_counter()
            print("\t\ttime passed: %d seconds" % (time_end - time_start))

        # Create a plot
        plt.figure(figsize=[13, 9.7])
        pi1_str = " - pi1: %.1f" % specific_pi1 if specific_pi1 is not None else ""
        title = "Polynomial SVM (d={} - c={}) (K: {:.1f}{}) - {}".format(d, c, K, pi1_str, conf.to_compact_string())
        plt.title(title)
        plt.xlabel("C")
        plt.ylabel("minDCF")
        plt.xscale('log')
        x = Cs
        for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
            y = minDCFs[:, app_i].flatten()
            plt.plot(x, y, label="minDCF (π=%.1f)" % pi1)
        plt.legend()
        if specific_pi1 is not None:
            pi1_without_points = "%.1f" % specific_pi1
            pi1_without_points = pi1_without_points.replace(".", "")
        pi1_str = "_train-pi1-%s" % pi1_without_points if specific_pi1 is not None else ""

        c_without_points = ("%.1f" % c).replace(".", "-")
        c_without_points = "_c-" + c_without_points

        d_str = "_d-%d" % d
        Kstr = "%.1f" % K
        Kstr = Kstr.replace(".", "-")
        Kstr = "K-" + Kstr

        if X_test is not None:
            if partial:
                base_path = EVAL_PARTIAL_POLYNOMIAL_SVM_GRAPH_PATH
            else:
                base_path = EVAL_FULL_POLYNOMIAL_SVM_GRAPH_PATH
        else:
            base_path = POLYNOMIAL_SVM_GRAPH_PATH

        full_path = "%s%s%s%s_%s%s" % (base_path, Kstr, d_str, c_without_points, conf.to_compact_string(), pi1_str)

        plt.savefig(full_path)
        print(f"Plot saved in {full_path}.")

    def polynomial_svm_gridsearch(X_train, y_train, X_test=None, y_test=None, partial=False):
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

        # Grid polynomial svm hyperparameters
        Ks = [1]
        Cs = np.logspace(-2, 1, 4)
        ds = [2, 3]
        cs = [1]

        # Grid search without class-balacing
        tot_time_start = time.perf_counter()
        print("Grid search on Polynomial SVM without class balancing started.")
        tot_iterations_required = len(preproc_configurations) * len(Ks) * len(ds) * len(cs) * len(Cs)
        tot_gs_iterations_required = len(preproc_configurations) * len(Ks) * len(ds) * len(cs)
        print("Total Polynomial SVM cross-validation required ", tot_iterations_required)
        print("Total grid search iterations required ", tot_gs_iterations_required)
        grid_search_iterations = 1
        for conf_i, conf in enumerate(preproc_configurations):
            for d in ds:
                for c in cs:
                    for Ki, K in enumerate(Ks):
                        print("Grid search iteration %d / %d" % (grid_search_iterations, tot_gs_iterations_required))
                        time_start = time.perf_counter()
                        plot_against_C(conf, K, Cs, d, c, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                       partial=partial, specific_pi1=None)
                        time_end = time.perf_counter()
                        grid_search_iterations += 1
                        print("Grid search iteration ended in %d seconds" % (time_end - time_start))
        tot_time_end = time.perf_counter()
        print("Grid search on Polynomial SVM without class balancing ended in %d seconds" % (tot_time_end - tot_time_start))

    def polynomial_svm_class_balancing(X_train, y_train, X_test=None, y_test=None):
        # We select the best preproc configuration, polynomial degree and C value
        preproc_conf = PreprocessConf([
                PreprocStage(Preproc.Centering),
                PreprocStage(Preproc.Whitening_Within_Covariance),
                PreprocStage(Preproc.L2_Normalization)
            ])
        K = 1
        d = 3
        C = 0.1

        kernel = SVM_Classifier.Kernel_Polynomial(d, 1)
        # Then, we try the best hyperparameters but now class-balancing with respect to the target application
        print("Trying the best hyperparameters but class-balancing w.r.t target applications..")
        for app_i, (train_pi1, Cfn, Cfp) in enumerate(applications):
            if X_test is None:
                print(f"Polynomial SVM cross-validation with class-balancing for the target application with π={train_pi1:.1f} (d={d} - c={1}) (C={C:.0e} - K={K:.1f}) - Preprocessing: {preproc_conf}")
            else:
                print(f"Polynomial SVM training on the train dataset and evaluating on the eval dataset with class-balancing for the target application with π={train_pi1:.1f} (d={d} - c={1}) (C={C:.0e} - K={K:.1f}) - Preprocessing: {preproc_conf}")

            time_start = time.perf_counter()
            scores, labels, _, _ = cross_validate_svm(preproc_conf, C, K, X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, X_val=None, y_val=None, specific_pi1=train_pi1, kernel=kernel)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
            time_end = time.perf_counter()
            print("Target application (π=%.1f) specific training cross-validation ended in %d seconds" % (
            pi1, (time_end - time_start)))
        print("Operation finished")

    # TRAIN
    # Grid search to select the best hyperparameters
    if args.gridsearch:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, POLYNOMIAL_SVM_TRAINLOG_FNAME)):
            print("Grid search Cross-Validation on the training dataset for Polynomial SVM")
            polynomial_svm_gridsearch(X_train=folds_data, y_train=folds_labels, X_test=None, y_test=None, partial=False)

    if args.class_balancing:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, POLYNOMIAL_SVM_CLASS_BALANCING_TRAINLOG_FNAME)):
            print("Cross-Validation on the training dataset for Polynomial SVM class-balancing with a prior")
            polynomial_svm_class_balancing(X_train=folds_data, y_train=folds_labels, X_test=None, y_test=None)

    # EVALUATION
    # PARTIAL TRAINING DATASET
    if args.eval_partial_gridsearch:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_POLYNOMIAL_SVM_TRAINLOG_FNAME)):
            print("Grid search training on partial training dataset and evaluating on the evaluation dataset for Polynomial SVM")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            polynomial_svm_gridsearch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, partial=True)

    if args.eval_partial_class_balancing:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_POLYNOMIAL_SVM_CLASS_BALANCING_TRAINLOG_FNAME)):
            print("Training on partial training dataset and evaluating on the evaluation datasetfor Polynomial SVM class-balancing with a prior")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            polynomial_svm_class_balancing(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # FULL TRAINING DATASET
    if args.eval_full_gridsearch:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_POLYNOMIAL_SVM_TRAINLOG_FNAME)):
            print(
                "Grid search training on the full training dataset and evaluating on the evaluation dataset for Polynomial SVM")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            polynomial_svm_gridsearch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, partial=False)

    if args.eval_full_class_balancing:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH,
                                             EVAL_FULL_POLYNOMIAL_SVM_CLASS_BALANCING_TRAINLOG_FNAME)):
            print(
                "Training on the full training dataset and evaluating on the evaluation datasetfor Polynomial SVM class-balancing with a prior")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            polynomial_svm_class_balancing(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


    plt.show()
