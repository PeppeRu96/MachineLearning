import matplotlib.pyplot as plt
import time

from wine_project.utility.ds_common import *
import evaluation.common as eval

from classifiers.logistic_regression import cross_validate_lr

SCRIPT_PATH = os.path.dirname(__file__)
TRAINLOGS_BASEPATH = os.path.join(SCRIPT_PATH, "..", "train_logs", "lr")
LINEAR_LR_TRAINLOG_FNAME = "linear_lr_trainlog_1.txt"
QUADRATIC_LR_TRAINLOG_FNAME = "quadratic_lr_trainlog_1.txt"
LINEAR_LR_BEST_TRAINLOG_FNAME = "linear_lr_best_trainlog_1.txt"
QUADRATIC_LR_BEST_TRAINLOG_FNAME = "quadratic_lr_best_trainlog_1.txt"
LINEAR_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "lr", "linear_lr_graph_")
QUADRATIC_LR_GRAPH_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "lr", "quadratic_lr_graph_")

# This script has to be called with one (or more) command-line parameters among
# {linear, quadratic, linear_best, quadratic_best}
# in order to start the requested analysis.
# It produces as output some plot graphs (one for each tried configuration)
# under the folder /graphs comparing minDCFs for different lambda regularizers
# for the different target applications. It also produces a training log
# for each analysis under the folder /train_logs
if __name__ == "__main__":
    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

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

    def lr_plot_against_lambda(conf, lambdas, specific_pi1=None, quadratic=False):
        """
        Given a specific preproc configuration, train n models for the different lambdas and compute minDCF for the apps.
        Then, display a plot for the different target applications.
        """
        minDCFs = np.zeros((len(lambdas), len(applications)))
        for i, l in enumerate(lambdas):
            print("\tLambda iteration ", i + 1)
            time_start = time.perf_counter()
            scores, labels = cross_validate_lr(folds_data, folds_labels, conf, l, pi1=specific_pi1, quadratic=quadratic)
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
        if (quadratic):
            path = QUADRATIC_LR_GRAPH_PATH
        else:
            path = LINEAR_LR_GRAPH_PATH
        if specific_pi1 is not None:
            pi1_without_points = "%.1f" % specific_pi1
            pi1_without_points = pi1_without_points.replace(".", "")

        pi1_str = "_train-pi1-%s" % pi1_without_points if specific_pi1 is not None else ""
        plt.savefig("%s%s%s" % (path, conf.to_compact_string(), pi1_str))

        return minDCFs

    def lr_gridsearch(specific_pi1=None, quadratic=False):
        pi1_str = "with prior weight specific training (π=%.1f)" % (specific_pi1) if specific_pi1 is not None else ""
        print("%s Logistic Regression %s analysis started" % ("Quadratic" if quadratic else "Linear", pi1_str))
        total_time_start = time.perf_counter()

        lambdas = np.logspace(-5, 3, 9)

        # Grid search
        print("Total LR cross-validation required ", len(preproc_configurations) * len(lambdas))
        grid_search_iterations = 1
        for conf_i, conf in enumerate(preproc_configurations):
            print("Grid search iteration ", grid_search_iterations)
            lr_plot_against_lambda(conf, lambdas, specific_pi1, quadratic)
            grid_search_iterations += 1

        total_time_end = time.perf_counter()
        print("LR analysis finished. Plots saved in %s. Total time: %d seconds.\n\n" %
              (QUADRATIC_LR_GRAPH_PATH if quadratic else LINEAR_LR_GRAPH_PATH, total_time_end - total_time_start))

    args = sys.argv[1:]

    if "linear" in args:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, LINEAR_LR_TRAINLOG_FNAME)):
            lr_gridsearch(quadratic=False)
    if "quadratic" in args:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, QUADRATIC_LR_TRAINLOG_FNAME)):
            lr_gridsearch(quadratic=True)

    # Best hyperparameters
    linear_preproc_conf = preproc_configurations[0] # TODO
    linear_lambda = 0.01 # TODO

    quadratic_preproc_conf = preproc_configurations[0]  # TODO
    quadratic_lambda = 0.01  # TODO

    # Train new models with the best hyper parameters but embedding a
    # specific prior = application prior in the training process
    lambdas = np.logspace(-5, 3, 9)
    # LINEAR LR
    if "linear_best" in args:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, LINEAR_LR_BEST_TRAINLOG_FNAME)):
            for (pi1, Cfn, Cfp) in applications:
                lr_plot_against_lambda(linear_preproc_conf, lambdas, pi1, False)

    # QUADRATIC LR
    if "quadratic_best" in args:
        with LoggingPrinter(incremental_path(TRAINLOGS_BASEPATH, QUADRATIC_LR_BEST_TRAINLOG_FNAME)):
            for (pi1, Cfn, Cfp) in applications:
                lr_plot_against_lambda(linear_preproc_conf, lambdas, pi1, True)

    plt.show()
