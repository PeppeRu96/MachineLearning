import numpy as np
import matplotlib.pyplot as plt

from preproc.dim_reduction.lda import lda
from preproc.dim_reduction.pca import pca
import preproc.dstools as dst
import wine_project.utility.ds_common as dsc
import evaluation.common as eval

from classifiers.gaussian_classifier import MVG_Classifier

if __name__ == "__main__":
    folds_data, folds_labels = dsc.load_train_dataset_5_folds()

    def cross_validate_MVG(gaussianize_features, PCA, naive, tied):
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

        print("5-Fold Cross-Validation %s %s MVG training on %s features %s started" % (naive_str, tied_str, gauss_str, PCA_str))
        iterations = 1
        scores = []
        labels = []
        for DTR, LTR, DTE, LTE in dst.kfold_generate(folds_data, folds_labels):
            # Preprocess data
            # TODO: Before PCA or Gaussianization?
            if PCA is not None:
                mu = DTR.mean(1)
                mu = mu.reshape(mu.size, 1)
                P, DTR, _ = pca(DTR, PCA)
                # Centering validation data
                DTE = DTE - mu
                DTE = P.T @ DTE
            if gaussianize_features:
                DTR = dst.gaussianize_features(DTR, DTR)
                DTE = dst.gaussianize_features(DTR, DTE)

            # Train
            mvg = MVG_Classifier()
            mvg.train(DTR, LTR, naive, tied)

            # Validate
            s = mvg.compute_binary_classifier_llr(DTE)

            # Collect scores and associated labels
            scores.append(s)
            labels.append(LTE)

            iterations += 1

        scores = np.array(scores)
        scores = scores.flatten()
        labels = np.array(labels)
        labels = labels.flatten()

        return scores, labels

    applications = dsc.applications
    gauss_grid = [False, True]
    PCA_grid = [None, 10, 9]
    naive_grid = [False, True]
    tied_grid = [False, True]


    minDCFs = np.zeros((len(gauss_grid), len(PCA_grid), len(naive_grid), len(tied_grid), len(applications)))
    # Grid search (exhaustive only because the process is speed enough to make
    # 24 iterations (note that we are not embedding any information about the application)
    # We don't need to retrain a new model when we change the application we want to target
    # Therefore, we can train the models and next compute the minDCF for the different applications
    iterations = 1
    for gi, g in enumerate(gauss_grid):
        for pi, p in enumerate(PCA_grid):
            for ni, n in enumerate(naive_grid):
                for ti, t in enumerate(tied_grid):
                    print("Training iteration ", iterations)
                    scores, labels = cross_validate_MVG(g, p, n, t)
                    iterations += 1
                    for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                        minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp, -20, 20, 1000)
                        print("Min DCF: ", minDCF)
                        minDCFs[gi, pi, ni, ti, app_i] = minDCF

    # Display a table
    print("")
    print("---------------------------------------------------------------------------------")
    print("{:>20}{:>20}{:>10}\t".format("Model", "Gaussianized", "PCA"), end="")
    for (pi1, Cfn, Cfp) in applications:
        print("{:>10}".format("π=%.1f" % pi1), end="")
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
