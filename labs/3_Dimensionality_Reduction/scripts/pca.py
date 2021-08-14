import os
import numpy as np
import matplotlib.pyplot as plt
import preproc.dstools as dst
from preproc.dim_reduction.pca import pca

SCRIPT_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join("..", "..", "..", "datasets", "iris", "iris.csv")
dsPath = os.path.join(SCRIPT_PATH, DATASET_PATH)

if (__name__ == "__main__"):
    # Start code
    ds = dst.load_iris_from_csv(dsPath)
    attributes = ds.samples
    labels = ds.labels

    P, DP, U = pca(attributes, 2)

    print("DP shape: ", DP.shape)

    print("Eigenvectors calculated:")
    print(U)

    Pld = np.load("../Solution/IRIS_PCA_matrix_m4.npy")
    print("\nEigenvectors from the solution:")
    print(Pld)

    print("\n2 principal components eigenvectors calculated:")
    print(P)

    plt.figure()
    plt.title("PCA")
    plt.xlabel("First direction")
    plt.ylabel("Second direction")
    for i, labelName in enumerate(ds.label_names):
        Mi = (labels == i)
        Dx = DP[0, Mi]
        Dy = DP[1, Mi]
        plt.scatter(Dx, Dy, label=labelName)
    plt.legend()
    plt.show()
