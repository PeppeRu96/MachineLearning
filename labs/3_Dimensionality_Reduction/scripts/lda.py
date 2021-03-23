import sys
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

sys.path.append('../../2_Iris/scripts/')

import iris_load_visualize as ilv

def sb_sw_compute(D, L):
    mu = D.mean(1)
    mu = mu.reshape(mu.size, 1)

    Sb = None
    Sw = None
    for i, labelName in enumerate(ilv.VAL_TO_LABELS):
        Di = D[:, (L == i)]
        mu_i = Di.mean(1)
        mu_i = mu_i.reshape(mu_i.size, 1)
        # Between-class covariance processing
        tmp = (mu_i - mu)
        bc = (tmp @ tmp.T) * Di.shape[1]
        if (i == 0):
            Sb = bc
        else:
            Sb = Sb + bc
        # Within-class covariance processing
        DCi = Di - mu_i
        Ci = (np.dot(DCi, DCi.T)) / DCi.shape[1]
        if (i == 0):
            Sw = Ci * Di.shape[1]
        else:
            Sw = Sw + Ci * Di.shape[1]
    Sb = Sb / D.shape[1]
    Sw = Sw / D.shape[1]

    return Sb, Sw

def lda(D, L, m, use_gen_eig):
    Sb, Sw = sb_sw_compute(D, L)

    print ("Sb calculated: ", Sb)
    print("\nSw calculated: ", Sw)

    # Generalized Eigenvalue problem
    U = None
    if (use_gen_eig):
        s, U = scipy.linalg.eigh(Sb, Sw)
        W = U[:, ::-1][:, 0:m]
        UW, _, _ = np.linalg.svd(W)
        U = UW[:, 0:m]
    else:
        U, s, _ = np.linalg.svd(Sw)
        P1 = np.dot( np.dot(U, np.diag(1.0/(s**0.5))), U.T)
        Sbt = np.dot( np.dot(P1, Sb), P1.T)
        s, P2 = np.linalg.eigh(Sbt)
        P2 = P2[:, ::-1][:, 0:m]
        U = np.dot(P1.T, P2)

    DP = np.dot(U.T, D)
    return U, DP

if (__name__ == "__main__"):
    attributes, labels = ilv.loadDataset(ilv.dsPath)
    P, DP = lda(attributes, labels, 2, 0)

    print("Eigenvectors calculated:")
    print(P)

    Pld = np.load("../Solution/IRIS_LDA_matrix_m2.npy")
    print("\nEigenvectors from the solution:")
    print(Pld)

    # Checking if subspaces of cols belonging to the computed P matrix and P solution matrix are equal
    print ("Checking LDA correctness against provided solution..")
    compound = np.linalg.svd(np.hstack([P, Pld]))[1]
    print("Singular values: ", np.linalg.svd(np.hstack([P, Pld]))[1])
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
    for i, labelName in enumerate(ilv.VAL_TO_LABELS):
        Mi = (labels == i)
        Dx = DP[0, Mi]
        Dy = DP[1, Mi]
        plt.scatter(Dx, Dy, label=labelName)
    plt.legend()
    plt.show()