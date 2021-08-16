import numpy as np

# Computes the gaussian distributions density of a numpy array x, given the mean mu and the variance var
def GAU_pdf(x, mu, var):
    y = np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
    return y

# Computes the logarithm of the gaussian distributions density of a numpy array x, given the mean mu and the variance var
def GAU_logpdf(x, mu, var):
    y = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - ((x - mu) ** 2) / (2 * var)
    return y

# Computes the logarithm of the multivariate gaussian distributions density of a np array x with shape (#dimensions, #samples)
# given the mean mu (#dimensions, 1) and the covariance matrix C (#dimensions, #dimensions)
def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    y = []
    for i in range(x.shape[1]):
        col = x[:, i].reshape(M, 1)
        ycol = - 0.5 * M * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * np.dot((col - mu).T,
                                                                                            np.dot(np.linalg.inv(C),
                                                                                                   (col - mu)))
        y.append(ycol)
    y = np.array(y)
    y = y.reshape(y.shape[0])
    return y
