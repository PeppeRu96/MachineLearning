import sys
import numpy as np

sys.path.append('../../2_Iris/scripts/')

import iris_load_visualize as ilv

def pca(D, m):
    mu = D.mean(1)
    mu = mu.reshape(mu.size, 1)
    # Centering dataset using broadcasting
    DC = D - mu

    # Covariance matrix
    C = (np.dot(DC, DC.T))/DC.shape[1]
    # Extracting eigenvectors and eigenvalues from a symmetric matrix
    s, U = np.linalg.eigh(C)

    # Extracting the m best eigenvectors (corresponding to the m greatest eigenvalues)
    P = U[:, ::-1][:, 0:m]

    # Projection of dataset to new directions
    DP = np.dot(P.T, DC)

    #print("Eigenvectors: ")
    #print(P)

    return P, DP

if (__name__ == "__main__"):
    attributes, labels = ilv.loadDataset(ilv.dsPath)
    P, DP = pca(attributes, 4)

    print("Eigenvectors calculated:")
    print(P)

    Pld = np.load("../Solution/IRIS_PCA_matrix_m4.npy")
    print("\nEigenvectors from the solution:")
    print(Pld)
