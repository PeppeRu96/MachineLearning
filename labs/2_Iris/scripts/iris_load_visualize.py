import os
import matplotlib.pyplot as plt
import preproc.dstools as dst

SCRIPT_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join("..", "..", "..", "datasets", "iris", "iris.csv")
dsPath = os.path.join(SCRIPT_PATH, DATASET_PATH)

if (__name__ == "__main__"):
    # Start code
    ds = dst.load_iris_from_csv(dsPath)
    attributes = ds.samples
    labels = ds.labels

    # Visualize code
    # Histograms
    ds.visualize_histogram(dst.VISUALIZE.Specified, [0])
    # Scatters
    ds.visualize_scatter(dst.VISUALIZE.Specified, [0, 1])

    # Computing statistics on dataset
    # Mean
    ds.mu = ds.samples.mean(1)

    # Center the dataset (subtract the mean of each attribute to each corresponding attribute's value)
    ds.mu = ds.mu.reshape(ds.mu.shape[0], 1)  # Make it a column vector

    print("Mean: ", ds.mu)

    # Broadcast
    ds.samples_centered = ds.samples - ds.mu

    # Histograms
    dst.visualize_histogram(ds.samples_centered, ds.labels, ds.feature_names, ds.label_names, dst.VISUALIZE.Specified, [0])
    plt.show()
