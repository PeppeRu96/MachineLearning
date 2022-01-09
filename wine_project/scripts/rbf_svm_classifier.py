import matplotlib.pyplot as plt
import time

import numpy as np

from wine_project.utility.ds_common import *
import evaluation.common as eval

from classifiers.svm import cross_validate_svm, SVM_Classifier

import argparse

TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "svm")
RBF_SVM_TRAINLOG_FNAME = "rbf_svm_trainlog_1.txt"
RBF_SVM_CLASS_BALANCING_TRAINLOG_FNAME = "rbf_svm_class_balancing_trainlog_1.txt"

RBF_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "rbf", "rbf_svm_graph_")

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch Logistic Regression classificator building",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gridsearch", type=bool, default=False,
                        help="Start gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--class_balancing", type=bool, default=False,
                        help="Start cross-validation to try class-balancing with the best hyperparameters")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    def plot_against_C_gamma(conf, K, Cs, gs, specific_pi1=None):

        pi1_str = "with prior weight specific training (π=%.1f)" % (specific_pi1) if specific_pi1 is not None else ""
        minDCFs = np.zeros((len(Cs), len(gs), len(applications)))
        for Ci, C in enumerate(Cs):
            for gi, g in enumerate(gs):
                kernel = SVM_Classifier.Kernel_RadialBasisFunction(g)
                print("\t(Ci: {}) - 5-Fold Cross-Validation RBF SVM (gamma={:.0e}) {} (C={:.0e} - K={:.1f}) - Preprocessing: {}".format(
                    Ci, g, pi1_str, C, K, conf))
                time_start = time.perf_counter()
                scores, labels = cross_validate_svm(folds_data, folds_labels, conf, C, K, specific_pi1=specific_pi1, kernel=kernel)
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
            title = "RBF SVM (K: {:.1f}{}) - {} - target-π={:.1f}".format(K, pi1_str, conf.to_compact_string(), pi1)
            plt.title(title)
            plt.xlabel("C")
            plt.ylabel("minDCF")
            plt.xscale('log')
            x = Cs

            # Plot only some important values of gamma
            if len(gs) > 3:
                gs_to_plot = []
                for g in gs:
                    g_log10 = np.log10(g)
                    if g_log10 >= -3 and g_log10 < 0 and float(g_log10).is_integer():
                        gs_to_plot.append(g)
                    if len(gs_to_plot) >= 3:
                        break
            else:
                gs_to_plot = gs

            for gi, g in enumerate(gs_to_plot):
                y = minDCFs[:, gi, app_i].flatten()
                gamma_str = f"{int(np.log10(g))}" if g.is_integer() else f"{np.log10(g):.1f}"
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
            plt.savefig("%s%s_%s%s%s" % (RBF_SVM_GRAPH_PATH, Kstr, conf.to_compact_string(), pi1_str, target_pi1_str))

    def rbf_svm_gridsearch():
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

        # Grid rbf svm hyperparameters
        Ks = [1]
        Cs = np.logspace(-3, 3, 7)
        gamma = np.logspace(-3, 3, 7)

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
                    plot_against_C_gamma(conf, K, Cs, gamma)
                    time_end = time.perf_counter()
                    grid_search_iterations += 1
                    print("Grid search iteration ended in %d seconds" % (time_end - time_start))
        tot_time_end = time.perf_counter()
        print("Grid search on RBF SVM without class balancing ended in %d seconds" % (tot_time_end - tot_time_start))

    def rbf_svm_class_balancing():
        # We select the best preproc configuration, gamma and C value
        preproc_conf = PreprocessConf([
                PreprocStage(Preproc.Centering),
                PreprocStage(Preproc.Whitening_Within_Covariance),
                PreprocStage(Preproc.L2_Normalization)
            ])
        K = 1
        g = 10**(-3)
        C = 0.1

        kernel = SVM_Classifier.Kernel_RadialBasisFunction(g)

        # Then, we try the best hyperparameters but now class-balancing with respect to the target application
        print("Trying the best hyperparameters but class-balancing w.r.t target applications..")
        for app_i, (train_pi1, Cfn, Cfp) in enumerate(applications):
            print(f"RBF SVM cross-validation with class-balancing for the target application with π={train_pi1:.1f} (gamma={g:.0e}) (C={C:.0e} - K={K:.1f}) - Preprocessing: {preproc_conf}")
            time_start = time.perf_counter()
            scores, labels = cross_validate_svm(folds_data, folds_labels, preproc_conf, C, K, specific_pi1=train_pi1, kernel=kernel)
            for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
            time_end = time.perf_counter()
            print("Target application (π=%.1f) specific training cross-validation ended in %d seconds" % (
            pi1, (time_end - time_start)))
        print("Operation finished")

    # Grid search to select the best hyperparameters
    if args.gridsearch:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_TRAINLOG_FNAME)):
            rbf_svm_gridsearch()

    if args.class_balancing:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_CLASS_BALANCING_TRAINLOG_FNAME)):
            rbf_svm_class_balancing()

    plt.show()
