import os
import matplotlib.pyplot as plt
import numpy as np

import preproc.dstools as dst
import wine_project.utility.ds_common as dsc
import seaborn as sns

SCRIPT_PATH = os.path.dirname(__file__)
RAW_HISTOGRAMS_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "raw")
GAUSSIANIZED_HISTOGRAMS_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "gaussianized")
GAUSSIANIZED_LABEL0_HISTOGRAMS_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "gaussianized_Hf")
GAUSSIANIZED_LABEL1_HISTOGRAMS_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "gaussianized_Ht")

CORRELATION_PATH = os.path.join(SCRIPT_PATH, "..", "graphs", "correlation_heatmap")

if __name__ == "__main__":
    # Load the train dataset in a wrapper of type Dataset to reuse useful utilities
    ds_train_wrapper = dsc.load_train_dataset()

    # Visualize some statistics on the training dataset
    ds_train_wrapper.visualize_statistics()

    # Visualize histograms of the raw features
    #ds_train_wrapper.visualize_histogram(bins=60, save_path=RAW_HISTOGRAMS_PATH)

    print("")

    # The features present very irregular distributions and very large outliers, we thus proceed to the Gaussianization
    # of the features
    ds_train_gaussianized_wrapper = ds_train_wrapper.gaussianize_features(ds_train_wrapper.samples)

    # Visualize some statistics on the training data gaussianized
    ds_train_gaussianized_wrapper.visualize_statistics()

    # Visualize histograms of the gaussianized features
    #ds_train_gaussianized_wrapper.visualize_histogram(bins=40, save_path=GAUSSIANIZED_HISTOGRAMS_PATH)

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
    D1_wrapper.labels = np.zeros(D1.shape[1])

    D0_wrapper.visualize_histogram(bins=40, save_path=GAUSSIANIZED_LABEL0_HISTOGRAMS_PATH)
    D1_wrapper.visualize_histogram(bins=40, save_path=GAUSSIANIZED_LABEL1_HISTOGRAMS_PATH)

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

    #visualize_correlation_matrices()
    plt.show()
