import matplotlib.pyplot as plt
import time

import numpy as np

from wine_project.utility.ds_common import *
import evaluation.common as eval

from density_estimation.gaussian_mixture_model import LBG_estimate
from classifiers.gmm_classifier import GMM_Classifier

TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "gmm")
GMM_TRAINLOG_FNAME = "gmm_trainlog_1.txt"

GMM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "gmm", "gmm_graph_")

if __name__ == "__main__":
    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    def cross_validate_gmm(preproc_conf, alpha, psi, diag_cov, tied_cov, max_components):
        diag_cov_str = "Diagonal" if diag_cov else ""
        tied_cov_str = "Tied" if tied_cov else ""
        print("\t5-Fold Cross-Validation {} {} GMM (components from 1 to {}) - Preprocessing: {}".format(
            diag_cov_str, tied_cov_str, max_components, preproc_conf))
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
                                       stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == max_components), verbose=0)
            gmms_h1 = LBG_estimate(DTR1, alpha, psi=psi, diag_cov=diag_cov, tied_cov=tied_cov,
                                       stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == max_components), verbose=0)
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

    def gmm_gridsearch():
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
        tot_iterations_required = len(preproc_configurations) * len(diags) * len(tieds) * len(comps)
        tot_gs_iterations_required = len(preproc_configurations) * len(diags) * len(tieds)
        print("Total GMM cross-validation required ", tot_iterations_required)
        print("Total grid search iterations required ", tot_gs_iterations_required)
        grid_search_iterations = 1
        for conf_i, conf in enumerate(preproc_configurations):
            for d_i, diag in enumerate(diags):
                for t_i, tied in enumerate(tieds):
                    print("Grid search iteration %d / %d" % (grid_search_iterations, tot_gs_iterations_required))
                    time_start = time.perf_counter()
                    scores, labels = cross_validate_gmm(conf, 0.1, 0.01, diag, tied, max_comps)
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
        np.save("temp.npy", minDCFs)
        # Plot
        #minDCFs = np.load("temp.npy")
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
                    plt.savefig("%s%s%s%s" % (GMM_GRAPH_PATH, pi1_str, diag_str, tied_str))

    with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, GMM_TRAINLOG_FNAME)):
        gmm_gridsearch()

    plt.show()
