import numpy as np

from preproc.dim_reduction.pca import pca
import preproc.dstools as dst
from wine_project.utility.ds_common import *
import evaluation.common as eval

from classifiers.gaussian_classifier import MVG_Classifier

import argparse

TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "mvg")
EVAL_TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "mvg", "eval")
MVG_TRAINLOG_FNAME = "mvg_trainlog_1.txt"
EVAL_MVG_TRAIN_PARTIAL_TRAINLOG_FNAME = "eval_mvg_partial_trainlog_1.txt"
EVAL_MVG_TRAIN_FULL_TRAINLOG_FNAME = "eval_mvg_full_trainlog_1.txt"

create_folder_if_not_exist(os.path.join(TRAINLOGS_BASEPATH, "dummy.txt"))
create_folder_if_not_exist(os.path.join(EVAL_TRAINLOGS_BASEPATH, "dummy.txt"))

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch Logistic Regression classificator building",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gridsearch", default=False, action="store_true",
                        help="Start gridsearch cross-validation on the training dataset for the mvg models")
    parser.add_argument("--eval_partial_gridsearch", default=False, action="store_true",
                        help="Start gridsearch on the evaluation dataset for the mvg models using 4/5 of the training dataset")
    parser.add_argument("--eval_full_gridsearch", default=False, action="store_true",
                        help="Start gridsearch on the evaluation dataset for the mvg models using the full training dataset")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load 5-folds already split training dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    # Load the test dataset
    X_test, y_test = load_dataset(train=False, only_data=True)

    def cross_validate_MVG(gaussianize_features, PCA, naive, tied, X_train, y_train, X_test=None, y_test=None):
        gauss_str = ""
        PCA_str = ""
        naive_str = ""
        tied_str = ""
        if gaussianize_features:
            gauss_str = "Gaussianized"
        if PCA is not None:
            PCA_str = "and after applying PCA=%d" % PCA
        if naive:
            naive_str = "Naive"
        if tied:
            tied_str = "Tied"

        def train_and_validate(DTR, LTR, DTE, LTE):
            # Preprocess data
            if gaussianize_features:
                DTRoriginal = np.array(DTR)
                DTR = dst.gaussianize_features(DTRoriginal, DTR)
                DTE = dst.gaussianize_features(DTRoriginal, DTE)
            if PCA is not None:
                mu = DTR.mean(1)
                mu = mu.reshape(mu.size, 1)
                P, DTR, _ = pca(DTR, PCA)
                # Centering validation data
                DTE = DTE - mu
                DTE = P.T @ DTE

            # Train
            mvg = MVG_Classifier(K=2)
            mvg.train(DTR, LTR, naive, tied)

            # Validate
            s = mvg.compute_binary_classifier_llr(DTE)

            return s

        if X_test is None:
            # Cross-validation
            print("5-Fold Cross-Validation %s %s MVG training on %s features %s started" % (naive_str, tied_str, gauss_str, PCA_str))
            iterations = 1
            scores = []
            labels = []
            for DTR, LTR, DTE, LTE in dst.kfold_generate(X_train, y_train):
                s = train_and_validate(DTR, LTR, DTE, LTE)

                # Collect scores and associated labels
                scores.append(s)
                labels.append(LTE)

                iterations += 1

            scores = np.array(scores)
            scores = scores.flatten()
            labels = np.array(labels)
            labels = labels.flatten()
        else:
            # Standard train-validation on fixed split
            print("Train and validation %s %s MVG training on %s features %s started" % (naive_str, tied_str, gauss_str, PCA_str))
            scores = train_and_validate(X_train, y_train, X_test, y_test)
            scores = scores.flatten()
            labels = y_test

        return scores, labels

    def mvg_gridsearch(X_train, y_train, X_test=None, y_test=None):
        gauss_grid = [False, True]
        PCA_grid = [None, 10, 9]
        naive_grid = [False, True]
        tied_grid = [False, True]

        if (X_test is not None):
            print(f"Training samples: {y_train.shape[0]}\nEval samples: {y_test.shape[0]}")

        tot_grid_iterations = len(gauss_grid) * len(PCA_grid) * len(naive_grid) * len(tied_grid)
        print(f"Grid search total iterations: {tot_grid_iterations}")


        minDCFs = np.zeros((len(gauss_grid), len(PCA_grid), len(naive_grid), len(tied_grid), len(applications)))
        # Grid search (exhaustive only because the process is speed enough to make
        # 24 iterations (note that we are not embedding any information about the application)
        # We don't need to retrain a new model when we change the application we want to target
        # Therefore, we can train the models and next compute the minDCF for the different applications
        it = 1
        for gi, g in enumerate(gauss_grid):
            for pi, p in enumerate(PCA_grid):
                for ni, n in enumerate(naive_grid):
                    for ti, t in enumerate(tied_grid):
                        print(f"Grid search iteration {it} / {tot_grid_iterations}")
                        scores, labels = cross_validate_MVG(g, p, n, t, X_train, y_train, X_test, y_test)
                        it += 1
                        for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                            minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                            print("Min DCF (π=%.2f) : %.3f " % (pi1, minDCF))
                            minDCFs[gi, pi, ni, ti, app_i] = minDCF

        # Display a table
        print("")
        print("---------------------------------------------------------------------------------")
        print("{:>20}{:>20}{:>10}\t".format("Model", "Gaussianized", "PCA"), end="")
        for (pi1, Cfn, Cfp) in applications:
            print("{:>10}".format("π=%.2f" % pi1), end="")
        print("")
        for gi, g in enumerate(gauss_grid):
            for pi, p in enumerate(PCA_grid):
                for ni, n in enumerate(naive_grid):
                    for ti, t in enumerate(tied_grid):
                        name = "-Cov"
                        if n:
                            name = "Diag" + name
                        else:
                            name = "Full" + name
                        if t:
                            name = "Tied " + name
                        if g:
                            gau_str = "Yes"
                        else:
                            gau_str = "No"
                        if p is not None:
                            pca_str = "%d" % p
                        else:
                            pca_str = "No"
                        print("{:>20}{:>20}{:>10}\t".format(name, gau_str, pca_str), end="")
                        for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                            print("{:>10.3f}".format(minDCFs[gi, pi, ni, ti, app_i]), end="")
                        print("")
        print("---------------------------------------------------------------------------------")

    # Grid search cross-validation on the training dataset
    if args.gridsearch:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, MVG_TRAINLOG_FNAME)):
            print("Grid search Cross-Validation on the training dataset")
            mvg_gridsearch(folds_data, folds_labels)

    # Grid search on the evaluation dataset using 4/5 of the training dataset
    if args.eval_partial_gridsearch:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_MVG_TRAIN_PARTIAL_TRAINLOG_FNAME)):
            print("Grid search on the evaluation dataset using 4/5 of the training dataset")
            X_train, y_train = concat_kfolds(folds_data[:-1], folds_labels[:-1])
            mvg_gridsearch(X_train, y_train, X_test, y_test)

    # Grid search on the evaluation dataset using 4/5 of the training dataset
    if args.eval_full_gridsearch:
        with LoggingPrinter(incremental_path(EVAL_TRAINLOGS_BASEPATH, EVAL_MVG_TRAIN_FULL_TRAINLOG_FNAME)):
            print("Grid search on the evaluation dataset using the full training dataset")
            X_train, y_train = concat_kfolds(folds_data, folds_labels)
            mvg_gridsearch(X_train, y_train, X_test, y_test)