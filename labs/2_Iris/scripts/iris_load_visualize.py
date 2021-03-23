import sys
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join("..", "..", "..", "datasets", "iris", "iris.csv")
LABELS_TO_VAL = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
VAL_TO_LABELS = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
VAL_TO_ATTRIBUTES = ["Sepal length", "Sepal width", "Petal length", "Petal width"]

dsPath = os.path.join(SCRIPT_PATH, DATASET_PATH)


def loadDataset(dsPath):
    f = open(dsPath, "r")
    attrList = []
    labels = []
    for line in f:
        if (len(line.split(",")) != 5):
            continue

        line = line.strip().split(",")
        attr = [float(a) for a in line[:-1]]
        label = LABELS_TO_VAL[line[-1]]

        attrList.append(attr)
        labels.append(label)

    f.close()
    return np.array(attrList).T, np.array(labels)


def visualizeHistograms(D, L):
    # For each attribute
    for idxAttr, nameAttr in enumerate(VAL_TO_ATTRIBUTES):
        # print("Showing histogram for attribute '%s'." % (nameAttr))
        plt.figure()
        plt.title(nameAttr)
        plt.xlabel(nameAttr)
        # For each label
        for i, labelName in enumerate(VAL_TO_LABELS):
            Mi = (L == i)  # Mask
            Di = D[
                idxAttr, Mi]  # Filter columns (selects only the cols, that are the samples which belong to the current label) and select the row of the curr attribute
            # print("Shape of data passed to histogram: ", D.shape)
            plt.hist(Di, density=True, label=labelName, alpha=0.5)

        plt.legend()


def visualizeScatters(D, L):
    for idxAttrX, nameAttrX in enumerate(VAL_TO_ATTRIBUTES):
        for idxAttrY, nameAttrY in enumerate(VAL_TO_ATTRIBUTES):
            if (idxAttrX == idxAttrY):
                continue
            plt.figure()
            plt.title("Scatter plot comparing %s to %s" % (nameAttrX, nameAttrY))
            plt.xlabel(nameAttrX)
            plt.ylabel(nameAttrY)
            for i, labelName in enumerate(VAL_TO_LABELS):
                Mi = (L == i)
                Dix = D[idxAttrX, Mi]
                Diy = D[idxAttrY, Mi]
                plt.scatter(Dix, Diy, label=labelName)
            plt.legend()

if (__name__ == "__main__"):
    # Start code
    attributes, labels = loadDataset(dsPath)
    # print("Attributes shape: ", attributes.shape, "\tLabels shape: ", labels.shape)

    # Visualize code
    # Histograms
    visualizeHistograms(attributes, labels)
    # Scatters
    visualizeScatters(attributes, labels)

    # Computing statistics on dataset
    # Mean
    mu = attributes.mean(1)
    print("Mean: ", mu)

    # Center the dataset (subtract the mean of each attribute to each corresponding attribute's value)
    mu = mu.reshape(mu.shape[0], 1)  # Make it a column vector
    # Broadcast
    attrsCentered = attributes - mu

    # Histograms
    visualizeHistograms(attrsCentered, labels)
    plt.show()
    pass