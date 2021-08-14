import numpy as np

# D: dataset samples
# m: number of the desired pca dimensions
# Returns P (linear transformation matrix with m columns), DP (projected dataset samples), U (pca transformation matrix with m=n desired dimension)
def pca(D, m):
    mu = D.mean(1)
    mu = mu.reshape(mu.size, 1)
    # Centering dataset using broadcasting
    DC = D - mu

    # Covariance matrix
    C = (np.dot(DC, DC.T))/DC.shape[1]
    # Extracting eigenvectors and eigenvalues from a symmetric matrix
    s, U = np.linalg.eigh(C)

    # Sort eigenvectors in descending order
    U = U[:, ::-1]

    # Extracting the m best eigenvectors (corresponding to the m greatest eigenvalues)
    P = U[:, 0:m]

    # Projection of dataset to new directions
    DP = np.dot(P.T, DC)

    #print("Eigenvectors: ")
    #print(P)

    return P, DP, U