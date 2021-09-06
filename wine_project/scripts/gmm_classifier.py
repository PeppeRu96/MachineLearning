import matplotlib.pyplot as plt
import time

import numpy as np

from wine_project.utility.ds_common import *
import evaluation.common as eval

from density_estimation.gaussian_mixture_model import LBG_estimate
from classifiers.gmm_classifier import GMM_Classifier

TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs")
GMM_TRAINLOG_FNAME = "gmm_trainlog_1.txt"

GMM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "linear_svm_graph_")

if __name__ == "__main__":
    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    def cross_validate_gmm(preproc_conf, alpha, psi, diag_cov, tied_cov, max_components):
        iterations = 1

        K = folds_data.shape[0]
        nk = folds_labels.shape[1]
        scores = np.zeros((max_components, K*nk))
        labels = np.zeros((max_components, K*nk))
        k = 0
        for DTR, LTR, DTE, LTE in dst.kfold_generate(folds_data, folds_labels):
            # Preprocess data
            DTR, DTE = preproc_conf.apply_preproc_pipeline(DTR, LTR, DTE)

            # Train all the gmms from 1 component to max_components components for class 0 and 1
            DTR0 = DTR[:, (LTR == 0)]
            DTR1 = DTR[:, (LTR == 1)]
            gmms_h0 = LBG_estimate(DTR0, alpha, psi=psi, diag_cov=diag_cov, tied_cov=tied_cov,
                                       stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == max_components), verbose=1)
            gmms_h1 = LBG_estimate(DTR1, alpha, psi=psi, diag_cov=diag_cov, tied_cov=tied_cov,
                                       stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == max_components), verbose=1)
            print("LBG estimated")
            gmm_classifiers = []
            for g0, g1 in zip(gmms_h0, gmms_h1):
                gmms = [g0, g1]
                gmm_classifiers.append(GMM_Classifier(gmms))

            # Now gmm_classifiers contains `max_components` gmm classifiers

            # Validate
            for i, gmm_classifier in enumerate(gmm_classifiers):
                s = gmm_classifier.compute_binary_llr(DTE)
                scores[i, k*nk:(k+1)*nk] = s
                labels[i, k*nk:(k+1)*nk] = LTE

            k += 1
            iterations += 1

        return scores, labels

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

    # Grid hyperparameters
    diags = [False, True]
    tieds = [False, True]
    max_comps = 256

    # Grid search
    tot_time_start = time.perf_counter()
    print("Grid search on GMM started.")
    tot_iterations_required = len(preproc_configurations) * len(diags) * len(tieds) * np.log2(max_comps)
    tot_gs_iterations_required = len(preproc_configurations) * len(diags) * len(tieds)
    print("Total GMM cross-validation required ", tot_iterations_required)
    print("Total grid search iterations required ", tot_gs_iterations_required)
    grid_search_iterations = 1
    for conf_i, conf in enumerate(preproc_configurations):
        for diag in diags:
            for tied in tieds:
                print("Grid search iteration %d / %d" % (grid_search_iterations, tot_gs_iterations_required))
                time_start = time.perf_counter()
                scores, labels = cross_validate_gmm(conf, 0.1, 0.01, diag, tied, max_comps)
                minDCFs = np.zeros((max_comps, len(applications)))
                for ci, c in enumerate(np.logspace(0, 8, 9, base=2)):
                    for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                        minDCF, _ = eval.bayes_min_dcf(scores[ci], labels[ci], pi1, Cfn, Cfp)
                        print("\t\tGMM with %d components min DCF (π=%.1f) : %.3f" % (c, pi1, minDCF))
                        minDCFs[ci, app_i] = minDCF

                time_end = time.perf_counter()
                grid_search_iterations += 1
                print("Grid search iteration ended in %d seconds" % (time_end - time_start))
    tot_time_end = time.perf_counter()
    print("Grid search on GMM without ended in %d seconds" % (tot_time_end - tot_time_start))
