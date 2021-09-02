import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from preproc.dim_reduction.pca import pca
import preproc.dstools as dst
import wine_project.utility.ds_common as dsc
import evaluation.common as eval

import time

from classifiers.logistic_regression import LogisticRegressionClassifier

SCRIPT_PATH = os.path.dirname(__file__)
LINEAR_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "linear_lr_graph_")
QUADRATIC_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "quadratic_lr_graph_")

if __name__ == "__main__":
    folds_data, folds_labels = dsc.load_train_dataset_5_folds()

    class configuration:
        def __init__(self, gaussianize, z_normalize, l2_normalize, whiten_covariance, whiten_within_covariance, pca):
            self.gaussianize = gaussianize
            self.z_normalize = z_normalize
            self.l2_normalize = l2_normalize
            self.whiten_covariance = whiten_covariance
            self.whiten_within_covariance = whiten_within_covariance
            self.pca = pca

        def to_string(self):
            g_str = "Gaussianization" if self.gaussianize else ""
            z_str = "Z-Normalization" if self.z_normalize else ""
            l2_str = "L2-Normalization" if self.l2_normalize else ""
            wc_str = "Whiten-Covariance" if self.whiten_covariance else ""
            wwc_str = "Whiten-Within-Covariance" if self.whiten_within_covariance else ""
            pca_str = "Pca: %d" % (self.pca) if self.pca is not None else ""


            return "%s %s %s %s %s %s" % (g_str, z_str, l2_str, wc_str, wwc_str, pca_str)

    def LR_analysis(quadratic=False):
        def cross_validate_LR(configuration, lambda_regularizer, pi1=None):
            gauss_str = ""
            z_normalize_str = ""
            l2_normalize_str = ""
            whiten_covariance_str = ""
            whiten_within_covariance_str = ""
            pca_str = ""
            if configuration.gaussianize:
                gauss_str = "Gaussianization"
            if configuration.z_normalize:
                z_normalize_str = "Z-Normalization"
            if configuration.whiten_covariance:
                whiten_covariance_str = "Whitening-Covariance"
            if configuration.whiten_within_covariance:
                whiten_within_covariance_str = "Whitening-Within-Class-Covariance"
            if configuration.l2_normalize:
                l2_normalize_str = "L2-Normalization"
            if configuration.pca is not None:
                pca_str = "PCA=%d" % configuration.pca
            preproc_str = "%s %s %s %s %s %s" % (gauss_str, z_normalize_str, l2_normalize_str, whiten_covariance_str, whiten_within_covariance_str, pca_str)

            print("\t\t5-Fold Cross-Validation Linear LR (λ=%.5f) - Preprocessing: %s" % (lambda_regularizer, preproc_str))
            iterations = 1
            scores = []
            labels = []
            for DTR, LTR, DTE, LTE in dst.kfold_generate(folds_data, folds_labels):
                # Preprocess data
                if configuration.gaussianize:
                    DTRoriginal = np.array(DTR)
                    DTR = dst.gaussianize_features(DTRoriginal, DTR)
                    DTE = dst.gaussianize_features(DTRoriginal, DTE)
                if configuration.z_normalize:
                    DTRoriginal = np.array(DTR)
                    DTR = dst.z_normalize(DTRoriginal, DTR)
                    DTE = dst.z_normalize(DTRoriginal, DTE)
                if configuration.l2_normalize:
                    DTR = dst.L2_normalize(DTR)
                    DTE = dst.L2_normalize(DTE)
                if configuration.whiten_covariance:
                    DTRoriginal = np.array(DTR)
                    DTR = dst.whiten_covariance_matrix(DTRoriginal, DTR)
                    DTE = dst.whiten_covariance_matrix(DTRoriginal, DTE)
                if configuration.whiten_within_covariance:
                    DTRoriginal = np.array(DTR)
                    DTR = dst.whiten_within_covariance_matrix(DTRoriginal, LTR, DTR)
                    DTE = dst.whiten_within_covariance_matrix(DTRoriginal, LTR, DTE)
                if configuration.pca is not None:
                    mu = DTR.mean(1)
                    mu = mu.reshape(mu.size, 1)
                    P, DTR, _ = pca(DTR, configuration.pca)
                    # Centering validation data
                    DTE = DTE - mu
                    DTE = P.T @ DTE


                # Train
                linear_lr = LogisticRegressionClassifier()
                if quadratic:
                    linear_lr.train(DTR, LTR, lambda_regularizer, pi1=pi1, expand_feature_space_func=LogisticRegressionClassifier.quadratic_feature_expansion)
                else:
                    linear_lr.train(DTR, LTR, lambda_regularizer, pi1=pi1)

                # Validate
                s = linear_lr.compute_binary_classifier_llr(DTE)

                # Collect scores and associated labels
                scores.append(s)
                labels.append(LTE)

                iterations += 1

            scores = np.array(scores)
            scores = scores.flatten()
            labels = np.array(labels)
            labels = labels.flatten()

            return scores, labels

        configurations = [
            configuration(False, # Gaussianize
                          False, # Z-Normalization
                          False, # L2-Normalization
                          False, # Whiten Covariance Matrix
                          False, # Whiten Within Covariance Matrix
                          None), # PCA
            configuration(True,  # Gaussianize
                          False,  # Z-Normalization
                          False,  # L2-Normalization
                          False,  # Whiten Covariance Matrix
                          False,  # Whiten Within Covariance Matrix
                          None),  # PCA
            configuration(True,  # Gaussianize
                          False,  # Z-Normalization
                          False,  # L2-Normalization
                          False,  # Whiten Covariance Matrix
                          False,  # Whiten Within Covariance Matrix
                          10),  # PCA
            configuration(False,  # Gaussianize
                          True,  # Z-Normalization
                          False,  # L2-Normalization
                          False,  # Whiten Covariance Matrix
                          False,  # Whiten Within Covariance Matrix
                          None),  # PCA
            configuration(False,  # Gaussianize
                          True,  # Z-Normalization
                          True,  # L2-Normalization
                          True,  # Whiten Covariance Matrix
                          False,  # Whiten Within Covariance Matrix
                          None),  # PCA
            configuration(False,  # Gaussianize
                          True,  # Z-Normalization
                          True,  # L2-Normalization
                          False,  # Whiten Covariance Matrix
                          True,  # Whiten Within Covariance Matrix
                          None)  # PCA
        ]

        lambdas = np.logspace(-5, 3, 9)

        applications = dsc.applications

        minDCFs = np.zeros((len(configurations), len(lambdas), len(applications)))

        print("Total LR cross-validation required ", len(configurations) * len(lambdas))
        print("Expected total time required: ", len(configurations) * len(lambdas) * 13, " minutes")
        grid_search_iterations = 1
        for conf_i, conf in enumerate(configurations):
            print("Grid search iteration ", grid_search_iterations)
            for i, l in enumerate(lambdas):
                print("\tLambda iteration ", i+1)
                time_start = time.perf_counter()
                scores, labels = cross_validate_LR(conf, l)
                for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                    minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                    print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                    minDCFs[conf_i, i, app_i] = minDCF
                time_end = time.perf_counter()
                print("\t\ttime passed: %d seconds" % (time_end-time_start))
            plt.figure()
            title = "%s LR - %s" % ("Quadratic" if quadratic else "Linear", conf.to_string())
            plt.title(title)
            plt.xlabel("λ")
            plt.ylabel("minDCF")
            plt.xscale('log')
            x = lambdas
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                y = minDCFs[conf_i, :, app_i].flatten()
                #f = interp1d(x, y)
                #xnew = np.logspace(lambdas.min(), lambdas.max(), len(lambdas) * 4)
                #plt.plot(x, y, 'o', xnew, f(xnew), '-')
                plt.plot(x, y, label="minDCF (π=%.1f)" % pi1)
            plt.legend()
            if (quadratic):
                path = QUADRATIC_LR_GRAPH_PATH
            else:
                path = LINEAR_LR_GRAPH_PATH
            plt.savefig("%s%s" % (path, conf.to_string()))

            grid_search_iterations += 1

    print("Linear Logistic Regression analysis started")
    total_time_start = time.perf_counter()
    LR_analysis(quadratic=False)
    total_time_end = time.perf_counter()
    print("Linear Logistic Regression analysis ended in %d seconds" % (total_time_end-total_time_start))
    print("")
    print("Quadratic Logistic Regression analysis started")
    total_time_start = time.perf_counter()
    LR_analysis(quadratic=True)
    total_time_end = time.perf_counter()
    print("Quadratic Logistic Regression analysis started in %d seconds" % (total_time_end-total_time_start))

    plt.show()