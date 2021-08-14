import os
import numpy as np
import matplotlib.pyplot as plt
import preproc.dstools as dst
from preproc.dim_reduction.lda import lda

SCRIPT_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join("..", "..", "..", "datasets", "iris", "iris.csv")
dsPath = os.path.join(SCRIPT_PATH, DATASET_PATH)

if (__name__ == "__main__"):
    # Start code
    ds = dst.load_iris_from_csv(dsPath)
    attributes = ds.samples
    labels = ds.labels
    m = 2

    P, DP = lda(attributes, labels, m, False)

    print("Eigenvectors calculated:")
    print(P)

    Pld = np.load("../Solution/IRIS_LDA_matrix_m2.npy")
    print("\nEigenvectors from the solution:")
    print(Pld)

    # Checking if subspaces of cols belonging to the computed P matrix and P solution matrix are equal
    print ("Checking LDA correctness against provided solution..")
    compound = np.linalg.svd(np.hstack([P, Pld]))[1]
    print("Singular values: ", compound)
    compound = [a for a in compound if (a > 1.0e-12)]
    print("Singular values greater than 0: ", len(compound))
    if (len(compound) > 2):
        print("Warning: the solution cols' subspace and processed cols' subspace don't match.")
    else:
        print("The processed LDA matrix is valid.")

    # Scattered plot of dataset projected in 2-LDA directions
    plt.figure()
    plt.title("LDA")
    plt.xlabel("First direction")
    plt.ylabel("Second direction")
    for i, labelName in enumerate(ds.label_names):
        Mi = (labels == i)
        Dx = DP[0, Mi]
        Dy = DP[1, Mi]
        plt.scatter(Dx, Dy, label=labelName)
    plt.legend()
    plt.show()