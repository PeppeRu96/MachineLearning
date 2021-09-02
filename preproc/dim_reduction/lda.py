import numpy as np
import scipy.linalg

def sb_sw_compute(D, L):
    mu = D.mean(1)
    mu = mu.reshape(mu.size, 1)

    labels = set(L)
    labelCnt = len(labels);

    Sb = None
    Sw = None
    for i in range(labelCnt):
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
        Ci = np.dot(DCi, DCi.T)
        if (i == 0):
            Sw = Ci
        else:
            Sw = Sw + Ci
    Sb = Sb / D.shape[1]
    Sw = Sw / D.shape[1]

    return Sb, Sw

def lda(D, L, m, use_gen_eig):
    Sb, Sw = sb_sw_compute(D, L)

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