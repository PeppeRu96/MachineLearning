import os
import matplotlib.pyplot as plt
import numpy as np

import preproc.dstools as dst
import wine_project.utility.ds_common as dsc
import seaborn as sns

import argparse

SCRIPT_PATH = os.path.dirname(__file__)
RAW_HISTOGRAMS_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "ds_analysis", "raw", "raw")
RAW_HISTOGRAMS_PATH_PER_CLASS = os.path.join(SCRIPT_PATH, "..", "graphs", "ds_analysis", "raw", "raw_per_class")
GAUSSIANIZED_HISTOGRAMS_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "ds_analysis", "gau", "gaussianized")
GAUSSIANIZED_HISTOGRAMS_PATH_PER_CLASS = os.path.join(SCRIPT_PATH, "..", "graphs", "ds_analysis", "gau", "gaussianized_per_class")
GAUSSIANIZED_LABEL0_HISTOGRAMS_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "ds_analysis", "gau", "gaussianized_Hf")
GAUSSIANIZED_LABEL1_HISTOGRAMS_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "ds_analysis", "gau", "gaussianized_Ht")

CORRELATION_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "ds_analysis", "correlation_heatmap")

dsc.create_folder_if_not_exist(os.path.join(SCRIPT_PATH, "..", "graphs", "ds_analysis", "dummy.txt"))
dsc.create_folder_if_not_exist(RAW_HISTOGRAMS_PATH)
dsc.create_folder_if_not_exist(GAUSSIANIZED_HISTOGRAMS_PATH)

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch dataset preprocessing",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--hist_raw", type=bool, default=False, help="Visualize and save a histogram for the raw features")
    parser.add_argument("--hist_gau", type=bool, default=False, help="Visualize and save a histogram for the Gaussianized features")
    parser.add_argument("--show_correlations", type=bool, default=False, help="Visualize and save correlation matrices")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load the train dataset in a wrapper of type Dataset to reuse useful utilities
    ds_train_wrapper = dsc.load_dataset(train=True)

    # Visualize some statistics on the training dataset
    ds_train_wrapper.visualize_statistics()

    # Visualize histograms of the raw features
    if args.hist_raw:
        ds_train_wrapper.visualize_histogram(bins=60, separate_classes=False, base_title="Raw", save_path=RAW_HISTOGRAMS_PATH)
        ds_train_wrapper.visualize_histogram(bins=60, separate_classes=True, base_title="Raw", save_path=RAW_HISTOGRAMS_PATH_PER_CLASS)

    print("")

    # The features present very irregular distributions and very large outliers, we thus proceed to the Gaussianization
    # of the features
    ds_train_gaussianized_wrapper = ds_train_wrapper.gaussianize_features(ds_train_wrapper.samples)

    # Visualize some statistics on the training data gaussianized
    ds_train_gaussianized_wrapper.visualize_statistics()

    # Visualize histograms of the gaussianized features
    if args.hist_gau:
        ds_train_gaussianized_wrapper.visualize_histogram(bins=40, separate_classes=False, base_title="Gaussianized", save_path=GAUSSIANIZED_HISTOGRAMS_PATH)
        ds_train_gaussianized_wrapper.visualize_histogram(bins=40, separate_classes=True, base_title="Gaussianized", save_path=GAUSSIANIZED_HISTOGRAMS_PATH_PER_CLASS)

    # Visualize histograms of the gaussianized features per class
    D = ds_train_gaussianized_wrapper.samples
    L = ds_train_gaussianized_wrapper.labels
    D0 = D[:, (L == 0)]
    D1 = D[:, (L == 1)]
    D0_wrapper = ds_train_gaussianized_wrapper.deep_copy()
    D0_wrapper.samples = D0
    D0_wrapper.labels = np.zeros(D0.shape[1])

    D1_wrapper = ds_train_gaussianized_wrapper.deep_copy()
    D1_wrapper.samples = D1
    D1_wrapper.labels = np.zeros(D1.shape[1]) + 1

    if args.hist_gau:
        D0_wrapper.visualize_histogram(bins=40, separate_classes=False, base_title="Gaussianized - Low quality wine (Hf)", save_path=GAUSSIANIZED_LABEL0_HISTOGRAMS_PATH)
        D1_wrapper.visualize_histogram(bins=40, separate_classes=False, base_title="Gaussianized - High quality wine (Ht)", save_path=GAUSSIANIZED_LABEL1_HISTOGRAMS_PATH)

    def visualize_correlation_matrices():
        corrM = dst.correlation_matrix(ds_train_gaussianized_wrapper.samples)
        plt.figure(figsize=[10, 7.5])
        plt.title("Correlation matrix for the whole dataset")
        sns.heatmap(corrM, annot=True, vmin=-1, vmax=1, center=0, cmap= 'Greys', linewidths=3, linecolor='black')
        plt.savefig(CORRELATION_PATH + "gaussianized_whole_dataset")

        D = ds_train_gaussianized_wrapper.samples
        L = ds_train_gaussianized_wrapper.labels

        # For label Hf (low quality)
        corrM0 = dst.correlation_matrix(D[:, (L==0)])
        plt.figure(figsize=[10, 7.5])
        plt.title("Correlation matrix for the label 0 (Low quality wine) samples")
        sns.heatmap(corrM0, annot=True, vmin=-1, vmax=1, center=0, cmap= 'Reds', linewidths=3, linecolor='black')
        plt.savefig(CORRELATION_PATH + "gaussianized_label0_dataset")

        # For label Ht (high quality)
        corrM1 = dst.correlation_matrix(D[:, (L == 1)])
        plt.figure(figsize=[10, 7.5])
        plt.title("Correlation matrix for the label 1 (High quality wine) samples")
        sns.heatmap(corrM1, annot=True, vmin=-1, vmax=1, center=0, cmap='Blues', linewidths=3, linecolor='black')
        plt.savefig(CORRELATION_PATH + "gaussianized_label1_dataset")

    if args.show_correlations:
        visualize_correlation_matrices()

    # Load the test dataset in a wrapper of type Dataset to reuse useful utilities
    ds_test_wrapper = dsc.load_dataset(train=False)

    # Visualize some statistics on the test dataset
    ds_test_wrapper.visualize_statistics()

    plt.show()
