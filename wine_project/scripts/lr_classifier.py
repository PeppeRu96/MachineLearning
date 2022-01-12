import matplotlib.pyplot as plt
import time

from wine_project.utility.ds_common import *
import evaluation.common as eval

from classifiers.logistic_regression import cross_validate_lr

import argparse

SCRIPT_PATH = os.path.dirname(__file__)
TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "lr")
LINEAR_LR_TRAINLOG_FNAME = "linear_lr_trainlog_1.txt"
QUADRATIC_LR_TRAINLOG_FNAME = "quadratic_lr_trainlog_1.txt"
LINEAR_LR_BEST_TRAINLOG_FNAME = "linear_lr_best_trainlog_1.txt"
QUADRATIC_LR_BEST_TRAINLOG_FNAME = "quadratic_lr_best_trainlog_1.txt"

LINEAR_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "lr", "linear_lr_graph_")
QUADRATIC_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "lr", "quadratic_lr_graph_")

EVAL_TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "lr", "eval")
EVAL_PARTIAL_LINEAR_LR_TRAINLOG_FNAME = "eval_partial_linear_lr_trainlog_1.txt"
EVAL_PARTIAL_QUADRATIC_LR_TRAINLOG_FNAME = "eval_partial_quadratic_lr_trainlog_1.txt"
EVAL_PARTIAL_LINEAR_LR_BEST_TRAINLOG_FNAME = "eval_partial_linear_lr_best_trainlog_1.txt"
EVAL_PARTIAL_QUADRATIC_LR_BEST_TRAINLOG_FNAME = "eval_partial_quadratic_lr_best_trainlog_1.txt"

EVAL_PARTIAL_LINEAR_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "lr", "eval", "eval_partial_linear_lr_graph_")
EVAL_PARTIAL_QUADRATIC_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "lr", "eval", "eval_partial_quadratic_lr_graph_")

EVAL_FULL_LINEAR_LR_TRAINLOG_FNAME = "eval_full_linear_lr_trainlog_1.txt"
EVAL_FULL_QUADRATIC_LR_TRAINLOG_FNAME = "eval_full_quadratic_lr_trainlog_1.txt"
EVAL_FULL_LINEAR_LR_BEST_TRAINLOG_FNAME = "eval_full_linear_lr_best_trainlog_1.txt"
EVAL_FULL_QUADRATIC_LR_BEST_TRAINLOG_FNAME = "eval_full_quadratic_lr_best_trainlog_1.txt"

EVAL_FULL_LINEAR_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "lr", "eval", "eval_full_linear_lr_graph_")
EVAL_FULL_QUADRATIC_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "lr", "eval", "eval_full_quadratic_lr_graph_")

create_folder_if_not_exist(os.path.join(TRAINLOGS_BASEPATH, "dummy.txt"))
create_folder_if_not_exist(LINEAR_LR_GRAPH_PATH)
create_folder_if_not_exist(os.path.join(EVAL_TRAINLOGS_BASEPATH, "dummy.txt"))
create_folder_if_not_exist(EVAL_FULL_LINEAR_LR_GRAPH_PATH)

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch Logistic Regression classificator building",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--linear", default=False, action="store_true",
                        help="Start cross-validation for the linear regularized logistic regression model")
    parser.add_argument("--quadratic", default=False, action="store_true",
                        help="Start cross-validation for the quadratic regularized logistic regression model")
    parser.add_argument("--linear_app_specific", default=False, action="store_true",
                        help="Start cross-validation for the linear regularized logistic regression model trained for the specific target applications")
    parser.add_argument("--quadratic_app_specific", default=False, action="store_true",
                        help="Start cross-validation for the quadratic regularized logistic regression model trained for the specific target applications")

    parser.add_argument("--eval_partial_linear", default=False, action="store_true",
                        help="Start evaluation gridsearch using partial training dataset for the linear regularized logistic regression model")
    parser.add_argument("--eval_partial_quadratic", default=False, action="store_true",
                        help="Start evaluation gridsearch using partial training dataset for the quadratic regularized logistic regression model")
    parser.add_argument("--eval_full_linear", default=False, action="store_true",
                        help="Start evaluation gridsearch using full training dataset for the linear regularized logistic regression model")
    parser.add_argument("--eval_full_quadratic", default=False, action="store_true",
                        help="Start evaluation gridsearch using full training dataset for the quadratic regularized logistic regression model")
    parser.add_argument("--eval_partial_linear_app_specific", default=False, action="store_true",
                        help="Start evaluation using partial training dataset for the linear regularized logistic regression model trained for the specific target applications")
    parser.add_argument("--eval_partial_quadratic_app_specific", default=False, action="store_true",
                        help="Start evaluation using partial training dataset for the quadratic regularized logistic regression model trained for the specific target applications")
    parser.add_argument("--eval_full_linear_app_specific", default=False, action="store_true",
                        help="Start evaluation using full training dataset for the linear regularized logistic regression model trained for the specific target applications")
    parser.add_argument("--eval_full_quadratic_app_specific", default=False, action="store_true",
                        help="Start evaluation using full training dataset for the quadratic regularized logistic regression model trained for the specific target applications")

    return parser.parse_args()

# It produces as output some plot graphs (one for each tried configuration) against lambda
# under the folder /graphs/lr comparing minDCFs for different lambda regularizers
# for the different target applications. It also produces a training log
# for each analysis under the folder /train_logs/lr
if __name__ == "__main__":
    args = get_args()

    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    # Load the test dataset
    X_test, y_test = load_dataset(train=False, only_data=True)

    # Define preprocessing configurations (to be cross-validated as different models)
    preproc_configurations = [
        PreprocessConf([]),
        PreprocessConf([DimReductionStage(DimRed.Pca, 10)]),
        PreprocessConf([PreprocStage(Preproc.Gaussianization)]),
        PreprocessConf([
            PreprocStage(Preproc.Gaussianization),
            DimReductionStage(DimRed.Pca, 10)
        ]),
        PreprocessConf([PreprocStage(Preproc.Z_Normalization)]),
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

    def lr_plot_against_lambda(conf, lambdas, X_train, y_train, X_test=None, y_test=None, partial=False, specific_pi1=None, quadratic=False):
        """
        Given a specific preproc configuration, train n models for the different lambdas and compute minDCF for the apps.
        Then, display a plot for the different target applications.
        """
        minDCFs = np.zeros((len(lambdas), len(applications)))
        for i, l in enumerate(lambdas):
            print("\tLambda iteration ", i + 1)
            time_start = time.perf_counter()
            scores, labels = cross_validate_lr(conf, l, X_train, y_train, X_test=X_test, y_test=y_test, pi1=specific_pi1, quadratic=quadratic)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                minDCFs[i, app_i] = minDCF
            time_end = time.perf_counter()
            print("\t\ttime passed: %d seconds" % (time_end - time_start))
        # Create a plot
        plt.figure(figsize=[13, 9.7])
        pi1_str = " - pi1: %.1f -" % specific_pi1 if specific_pi1 is not None else ""
        title = "%s LR%s %s" % ("Quadratic" if quadratic else "Linear", pi1_str, conf.to_compact_string())
        plt.title(title)
        plt.xlabel("λ")
        plt.ylabel("minDCF")
        plt.xscale('log')
        x = lambdas
        for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
            y = minDCFs[:, app_i].flatten()
            plt.plot(x, y, label="minDCF (π=%.1f)" % pi1)
        plt.legend()

        if X_test is not None:
            if partial:
                if (quadratic):
                    path = EVAL_PARTIAL_QUADRATIC_LR_GRAPH_PATH
                else:
                    path = EVAL_PARTIAL_LINEAR_LR_GRAPH_PATH
            else:
                if (quadratic):
                    path = EVAL_FULL_QUADRATIC_LR_GRAPH_PATH
                else:
                    path = EVAL_FULL_LINEAR_LR_GRAPH_PATH
        else:
            if (quadratic):
                path = QUADRATIC_LR_GRAPH_PATH
            else:
                path = LINEAR_LR_GRAPH_PATH

        if specific_pi1 is not None:
            pi1_without_points = "%.1f" % specific_pi1
            pi1_without_points = pi1_without_points.replace(".", "")

        pi1_str = "_train-pi1-%s" % pi1_without_points if specific_pi1 is not None else ""

        full_path = "%s%s%s" % (path, conf.to_compact_string(), pi1_str)
        plt.savefig(full_path)

        print(f"Plot saved in {full_path}.")

        return minDCFs

    def lr_gridsearch(X_train, y_train, X_test=None, y_test=None, partial=False, specific_pi1=None, quadratic=False):
        """
        It produces as output some plot graphs (one for each tried configuration) against lambda
        under the folder /graphs/lr comparing minDCFs for different lambda regularizers
        for the different target applications. It also produces a training log
        for each analysis under the folder /train_logs/lr
        It produces also for evaluation in /graphs/lr/eval and /train_logs/lr/eval
        """
        pi1_str = "with prior weight specific training (π=%.1f)" % (specific_pi1) if specific_pi1 is not None else ""
        print("%s Logistic Regression %s analysis started" % ("Quadratic" if quadratic else "Linear", pi1_str))
        total_time_start = time.perf_counter()

        lambdas = np.logspace(-5, 3, 9)

        # Grid search
        print("Total LR cross-validation required ", len(preproc_configurations) * len(lambdas))
        grid_search_iterations = 1
        for conf_i, conf in enumerate(preproc_configurations):
            print("Grid search iteration ", grid_search_iterations)
            lr_plot_against_lambda(conf, lambdas, X_train, y_train, X_test=X_test, y_test=y_test, partial=partial, specific_pi1=specific_pi1, quadratic=quadratic)
            grid_search_iterations += 1

        total_time_end = time.perf_counter()
        print(f"LR gridsearch finished. Total time: {total_time_end - total_time_start} seconds.\n\n")

    # Grid search cross-validation on the training dataset for model selection and hyperparameters optimization
    if args.linear:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, LINEAR_LR_TRAINLOG_FNAME)):
            print("Grid search Cross-Validation on the training dataset for Linear Regularized Logistic Regression")
            lr_gridsearch(X_train=folds_data, y_train=folds_labels, X_test=None, y_test=None, partial=False,\
                          specific_pi1=applications[0][0], quadratic=False)
    if args.quadratic:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, QUADRATIC_LR_TRAINLOG_FNAME)):
            print("Grid search Cross-Validation on the training dataset for Quadratic Regularized Logistic Regression")
            lr_gridsearch(X_train=folds_data, y_train=folds_labels, X_test=None, y_test=None, partial=False, \
                          specific_pi1=applications[0][0], quadratic=True)

    # Grid search on the evaluation dataset using K-1 folds of the training dataset (Linear LR)
    if args.eval_partial_linear:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_LINEAR_LR_TRAINLOG_FNAME)):
            print("Grid search on the evaluation dataset using 4/5 of the training dataset for Linear Regularized Logistic Regression")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            lr_gridsearch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, partial=True, \
                          specific_pi1=applications[0][0], quadratic=False)
    # Grid search on the evaluation dataset using K-1 folds of the training dataset (Quadratic LR)
    if args.eval_partial_quadratic:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_QUADRATIC_LR_TRAINLOG_FNAME)):
            print("Grid search on the evaluation dataset using 4/5 of the training dataset for Quadratic Regularized Logistic Regression")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            lr_gridsearch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, partial=True, \
                          specific_pi1=applications[0][0], quadratic=True)
    # Grid search on the evaluation dataset using the full training dataset (Linear LR)
    if args.eval_full_linear:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_LINEAR_LR_TRAINLOG_FNAME)):
            print("Grid search on the evaluation dataset using the full training dataset for Linear Regularized Logistic Regression")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            lr_gridsearch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, partial=False, \
                          specific_pi1=applications[0][0], quadratic=False)
    # Grid search on the evaluation dataset using the full training dataset (Quadratic LR)
    if args.eval_full_quadratic:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_QUADRATIC_LR_TRAINLOG_FNAME)):
            print(
                "Grid search on the evaluation dataset using the full training dataset for Quadratic Regularized Logistic Regression")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            lr_gridsearch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, partial=False, \
                          specific_pi1=applications[0][0], quadratic=True)

    # Best hyperparameters from hyperparameters optimization and model selection on the training dataset
    # Plug manually the best preprocess configuration and the best lambda to try training for application specific scenarios
    # Linear
    # Pi = 0.1
    linear_preproc_conf_pi01 = preproc_configurations[2]
    linear_lambda_pi01 = 0.01
    # Pi = 0.9
    linear_preproc_conf_pi09 = preproc_configurations[5]
    linear_lambda_pi09 = 0.00001

    # Quadratic
    quadratic_preproc_conf_pi01 = preproc_configurations[2]
    quadratic_lambda_pi01 = 0.001
    # Pi = 0.9
    quadratic_preproc_conf_pi09 = preproc_configurations[6]
    quadratic_lambda_pi09 = 0.01

    # Train new models with the best hyper parameters but embedding a
    # specific prior = application prior in the training process
    def _cross_validate_lr(conf, l, X_train, y_train, X_test=None, y_test=None, pi1=None, quadratic=False):
        time_start = time.perf_counter()
        scores, labels = cross_validate_lr(conf, l, X_train, y_train, X_test=X_test, y_test=y_test, pi1=pi1, quadratic=quadratic)
        minDCFs = []
        for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
            minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
            print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
            minDCFs.append(minDCF)
        time_end = time.perf_counter()
        print("\t\ttime passed: %d seconds" % (time_end - time_start))

        return minDCFs

    # TRAIN DATASET - LINEAR LR
    if args.linear_app_specific:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, LINEAR_LR_BEST_TRAINLOG_FNAME)):
            print(f"(TRAIN DATASET) Cross-validating Linear Logistic Regression with the best preproc configuration and the best lambda embedding target application prior in the training process..")
            pi01 = applications[1][0]
            print(f"\tLinear Logistic Regression training for target application specific π={pi01:.1f}..")
            _cross_validate_lr(linear_preproc_conf_pi01, linear_lambda_pi01, X_train=folds_data, y_train=folds_labels,
                               X_test=None, y_test=None, pi1=pi01, quadratic=False)

            pi09 = applications[2][0]
            print(f"\tLinear Logistic Regression training for target application specific π={pi09:.1f}..")
            _cross_validate_lr(linear_preproc_conf_pi09, linear_lambda_pi09, X_train=folds_data, y_train=folds_labels,
                               X_test=None, y_test=None, pi1=pi09, quadratic=False)

    print("\n")

    # TRAIN DATASET - QUADRATIC LR
    if args.quadratic_app_specific:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, QUADRATIC_LR_BEST_TRAINLOG_FNAME)):
            print(f"(TRAIN DATASET) Cross-validating Quadratic Logistic Regression with the best preproc configuration and the best lambda embedding target application prior in the training process..")
            pi01 = applications[1][0]
            print(f"\tQuadratic Logistic Regression training for target application specific π={pi01:.1f}..")
            _cross_validate_lr(quadratic_preproc_conf_pi01, quadratic_lambda_pi01, X_train=folds_data, y_train=folds_labels,
                               X_test=None, y_test=None, pi1=pi01, quadratic=True)

            pi09 = applications[2][0]
            print(f"\tQuadratic Logistic Regression training for target application specific π={pi09:.1f}..")
            _cross_validate_lr(quadratic_preproc_conf_pi09, quadratic_lambda_pi09, X_train=folds_data, y_train=folds_labels,
                               X_test=None, y_test=None, pi1=pi09, quadratic=True)

    # --------------------------------------------------------- #

    # EVALUATION DATASET - PARTIAL - LINEAR LR
    if args.eval_partial_linear_app_specific:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_LINEAR_LR_BEST_TRAINLOG_FNAME)):
            print(f"(EVALUATION DATASET - Partial Training Dataset) Training and validate Linear Logistic Regression with the best preproc configuration and the best lambda embedding target application prior in the training process..")
            pi01 = applications[1][0]
            print(f"\tLinear Logistic Regression training for target application specific π={pi01:.1f}..")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            _cross_validate_lr(linear_preproc_conf_pi01, linear_lambda_pi01, X_train=X_train, y_train=y_train,
                               X_test=X_test, y_test=y_test, pi1=pi01, quadratic=False)

            pi09 = applications[2][0]
            print(f"\tLinear Logistic Regression training for target application specific π={pi09:.1f}..")
            _cross_validate_lr(linear_preproc_conf_pi09, linear_lambda_pi09, X_train=X_train, y_train=y_train,
                               X_test=X_test, y_test=y_test, pi1=pi09, quadratic=False)

    print("\n")

    # EVALUATION DATASET - PARTIAL - QUADRATIC LR
    if args.eval_partial_quadratic_app_specific:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_PARTIAL_QUADRATIC_LR_BEST_TRAINLOG_FNAME)):
            print(
                f"(EVALUATION DATASET - Partial Training Dataset) Training and validate Quadratic Logistic Regression with the best preproc configuration and the best lambda embedding target application prior in the training process..")
            pi01 = applications[1][0]
            print(f"\tQuadratic Logistic Regression training for target application specific π={pi01:.1f}..")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            _cross_validate_lr(quadratic_preproc_conf_pi01, quadratic_lambda_pi01, X_train=X_train, y_train=y_train,
                               X_test=X_test, y_test=y_test, pi1=pi01, quadratic=True)

            pi09 = applications[2][0]
            print(f"\tQuadratic Logistic Regression training for target application specific π={pi09:.1f}..")
            _cross_validate_lr(quadratic_preproc_conf_pi09, quadratic_lambda_pi09, X_train=X_train, y_train=y_train,
                               X_test=X_test, y_test=y_test, pi1=pi09, quadratic=True)

    # EVALUATION DATASET - FULL - LINEAR LR
    if args.eval_full_linear_app_specific:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_LINEAR_LR_BEST_TRAINLOG_FNAME)):
            print(f"(EVALUATION DATASET - Full Training Dataset) Training and validate Linear Logistic Regression with the best preproc configuration and the best lambda embedding target application prior in the training process..")
            pi01 = applications[1][0]
            print(f"\tLinear Logistic Regression training for target application specific π={pi01:.1f}..")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            _cross_validate_lr(linear_preproc_conf_pi01, linear_lambda_pi01, X_train=X_train, y_train=y_train,
                               X_test=X_test, y_test=y_test, pi1=pi01, quadratic=False)

            pi09 = applications[2][0]
            print(f"\tLinear Logistic Regression training for target application specific π={pi09:.1f}..")
            _cross_validate_lr(linear_preproc_conf_pi09, linear_lambda_pi09, X_train=X_train, y_train=y_train,
                               X_test=X_test, y_test=y_test, pi1=pi09, quadratic=False)

    print("\n")

    # EVALUATION DATASET - FULL - QUADRATIC LR
    if args.eval_full_quadratic_app_specific:
        with LoggingPrinter(
                incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_FULL_QUADRATIC_LR_BEST_TRAINLOG_FNAME)):
            print(
                f"(EVALUATION DATASET - Full Training Dataset) Training and validate Quadratic Logistic Regression with the best preproc configuration and the best lambda embedding target application prior in the training process..")
            pi01 = applications[1][0]
            print(f"\tQuadratic Logistic Regression training for target application specific π={pi01:.1f}..")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            _cross_validate_lr(quadratic_preproc_conf_pi01, quadratic_lambda_pi01, X_train=X_train, y_train=y_train,
                               X_test=X_test, y_test=y_test, pi1=pi01, quadratic=True)

            pi09 = applications[2][0]
            print(f"\tQuadratic Logistic Regression training for target application specific π={pi09:.1f}..")
            _cross_validate_lr(quadratic_preproc_conf_pi09, quadratic_lambda_pi09, X_train=X_train, y_train=y_train,
                               X_test=X_test, y_test=y_test, pi1=pi09, quadratic=True)

    plt.show()
