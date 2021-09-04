import matplotlib.pyplot as plt
import time

from wine_project.utility.ds_common import *
import evaluation.common as eval

from classifiers.svm import SVM_Classifier

TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs")
LINEAR_SVM_TRAINLOG_FNAME = "linear_svm_trainlog_1.txt"

LINEAR_SVM_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "linear_svm_graph_")

if __name__ == "__main__":
    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    def cross_validate_svm(preproc_conf, C, K, specific_pi1=None, kernel=None, maxfun=15000, maxiter=15000, factr=1.0):
        iterations = 1
        scores = []
        labels = []
        for DTR, LTR, DTE, LTE in dst.kfold_generate(folds_data, folds_labels):
            # Preprocess data
            DTR, DTE = preproc_conf.apply_preproc_pipeline(DTR, LTR, DTE)

            # Train
            svm = SVM_Classifier()
            svm.train(DTR, LTR, C, K, specific_pi1, kernel, maxfun, maxiter, factr)

            # Validate
            s = svm.compute_scores(DTE)

            # Collect scores and associated labels
            scores.append(s)
            labels.append(LTE)

            iterations += 1

        scores = np.array(scores).flatten()
        labels = np.array(labels).flatten()

        return scores, labels

    def linear_svm_gridsearch():
        # preproc_configurations = [
        #     PreprocessConf([]),
        #     PreprocessConf([PreprocStage(Preproc.Gaussianization)]),
        #     PreprocessConf([
        #         PreprocStage(Preproc.Centering),
        #         PreprocStage(Preproc.Whitening_Covariance),
        #         PreprocStage(Preproc.L2_Normalization)
        #     ]),
        #     PreprocessConf([
        #         PreprocStage(Preproc.Centering),
        #         PreprocStage(Preproc.Whitening_Within_Covariance),
        #         PreprocStage(Preproc.L2_Normalization)
        #     ]),
        # ]

        preproc_configurations = [
            PreprocessConf([]),
            PreprocessConf([PreprocStage(Preproc.Gaussianization)])
        ]

        # Grid
        Ks = [1, 10]
        Cs = np.logspace(-2, 1, 4)

        def plot_against_C(conf, K, Cs, specific_pi1=None):
            pi1_str = "with prior weight specific training (π=%.1f)" % (specific_pi1) if specific_pi1 is not None else ""
            minDCFs = np.zeros((len(Cs), len(applications)))
            for Ci, C in enumerate(Cs):
                print("\t(Ci: {}) - 5-Fold Cross-Validation Linear SVM {} (C={:.0e} - K={:.1f}) - Preprocessing: {}".format(Ci, pi1_str, C, K, conf))
                time_start = time.perf_counter()
                scores, labels = cross_validate_svm(conf, C, K, specific_pi1=specific_pi1)
                for app_i, (pi1, Cfn, Cfp) in enumerate(applications):
                    minDCF, _ = eval.bayes_min_dcf(scores, labels, pi1, Cfn, Cfp)
                    print("\t\tmin DCF (π=%.1f) : %.3f" % (pi1, minDCF))
                    minDCFs[Ci, app_i] = minDCF
                time_end = time.perf_counter()
                print("\t\ttime passed: %d seconds" % (time_end - time_start))

            # Create a plot
            plt.figure(figsize=[13, 9.7])
            pi1_str = " - pi1: %.1f" % (specific_pi1) if specific_pi1 is not None else ""
            title = "Linear SVM (K: {:.1f}{}) - {}".format(K, pi1_str, conf.to_compact_string())
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
            pi1_str = "_train-pi1-%s" % (pi1_without_points) if specific_pi1 is not None else ""
            Kstr = "%.1f" % K
            Kstr = Kstr.replace(".", "-")
            Kstr = "K-" + Kstr
            plt.savefig("%s%s_%s%s" % (LINEAR_SVM_GRAPH_PATH, Kstr, conf.to_compact_string(), pi1_str))

        # Grid search without class-balacing
        tot_time_start = time.perf_counter()
        print("Grid search on Linear SVM without class balancing started.")
        tot_iterations_required = len(preproc_configurations) * len(Ks) * len(Cs)
        tot_gs_iterations_required = len(preproc_configurations) * len(Ks)
        print("Total Linear SVM cross-validation required ", tot_iterations_required)
        print("Total grid search iterations required ", tot_gs_iterations_required)
        grid_search_iterations = 1
        for conf_i, conf in enumerate(preproc_configurations):
            for Ki, K in enumerate(Ks):
                print("Grid search iteration %d / %d" % (grid_search_iterations, tot_gs_iterations_required))
                time_start = time.perf_counter()
                plot_against_C(conf, K, Cs)
                time_end = time.perf_counter()
                grid_search_iterations += 1
                print("Grid search iteration ended in %d seconds" % (time_end - time_start))
        tot_time_end = time.perf_counter()
        print("Grid search on Linear SVM without class balancing ended in %d seconds" % (tot_time_end - tot_time_start))

    args = sys.argv[1:]

    if "linear" in args:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, LINEAR_SVM_TRAINLOG_FNAME)):
            linear_svm_gridsearch()

    plt.show()
