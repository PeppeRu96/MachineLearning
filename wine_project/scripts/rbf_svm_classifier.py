import matplotlib.pyplot as plt
import time

import numpy as np

from wine_project.utility.ds_common import *
import evaluation.common as eval

from classifiers.svm import cross_validate_svm, SVM_Classifier

import argparse

TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "rbf", "svm")
RBF_SVM_TRAINLOG_FNAME = "rbf_svm_trainlog_1.txt"
RBF_SVM_FINE_GRAINED_TRAINLOG_FNAME = "rbf_svm_fine_grained_trainlog_1.txt"
RBF_SVM_CLASS_BALANCING_PI05_TRAINLOG_FNAME = "rbf_svm_class_balancing_pi05_trainlog_1.txt"
RBF_SVM_CLASS_BALANCING_PI01_TRAINLOG_FNAME = "rbf_svm_class_balancing_pi01_trainlog_1.txt"
RBF_SVM_CLASS_BALANCING_PI09_TRAINLOG_FNAME = "rbf_svm_class_balancing_pi09_trainlog_1.txt"

RBF_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "svm", "rbf", "rbf_svm_graph_")

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch Logistic Regression classificator building",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gridsearch", type=bool, default=False,
                        help="Start a coarse-level gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--gridsearch_fine_grained", type=bool, default=False,
                        help="Start a fine-grained gridsearch cross-validation to jointly optimize C and gamma for different preprocess configurations")
    parser.add_argument("--class_balancing", type=bool, default=False,
                        help="Start cross-validation to try class-balancing with the best hyperparameters")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    def plot_against_C_gamma(conf, K, Cs, gs, specific_pi1=None, prefix_title=""):

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
            plt.savefig("%s%s_%s%s%s_%s" % (RBF_SVM_GRAPH_PATH, Kstr, conf.to_compact_string(), pi1_str, target_pi1_str, prefix_title))

    def rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, prefix_title=""):
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
                    plot_against_C_gamma(conf, K, Cs, gamma, prefix_title=prefix_title)
                    time_end = time.perf_counter()
                    grid_search_iterations += 1
                    print("Grid search iteration ended in %d seconds" % (time_end - time_start))
        tot_time_end = time.perf_counter()
        print("Grid search on RBF SVM without class balancing ended in %d seconds" % (tot_time_end - tot_time_start))

    def rbf_svm_class_balancing(preproc_conf, K, g, C):
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
            train_pi1, (time_end - time_start)))
        print("Operation finished")


    # Coarse-level grid search to select the best hyperparameters
    if args.gridsearch:
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
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_TRAINLOG_FNAME)):
            rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma)

    # ----------------------------------------------------------------------- #

    # Fine-grained grid search to select the best hyperparameters
    if args.gridsearch_fine_grained:
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
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_FINE_GRAINED_TRAINLOG_FNAME)):
            rbf_svm_gridsearch(preproc_configurations, Ks, Cs, gamma, prefix_title="fine-grained")

    # -------------------------------------------------------------------------- #

    if args.class_balancing:
        # pi05 best model - select the best preproc configuration, gamma and C value
        preproc_conf = PreprocessConf([
            PreprocStage(Preproc.Centering),
            PreprocStage(Preproc.Whitening_Within_Covariance),
            PreprocStage(Preproc.L2_Normalization)
        ])
        K = 1
        g = 8
        C = 0.5
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_CLASS_BALANCING_PI05_TRAINLOG_FNAME)):
            rbf_svm_class_balancing(preproc_conf, K, g, C)

        # pi01 best model - select the best preproc configuration, gamma and C value
        preproc_conf = PreprocessConf([
            PreprocStage(Preproc.Centering),
            PreprocStage(Preproc.Whitening_Within_Covariance),
            PreprocStage(Preproc.L2_Normalization)
        ])
        K = 1
        g = 10
        C = 0.1
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_CLASS_BALANCING_PI01_TRAINLOG_FNAME)):
            rbf_svm_class_balancing(preproc_conf, K, g, C)

        # pi09 best model - select the best preproc configuration, gamma and C value
        preproc_conf = PreprocessConf([
            PreprocStage(Preproc.Centering),
            PreprocStage(Preproc.Whitening_Within_Covariance),
            PreprocStage(Preproc.L2_Normalization)
        ])
        K = 1
        g = 10
        C = 0.1
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, RBF_SVM_CLASS_BALANCING_PI09_TRAINLOG_FNAME)):
            rbf_svm_class_balancing(preproc_conf, K, g, C)

    plt.show()
