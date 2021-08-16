import numpy as np
import matplotlib.pyplot as plt
import classifiers.distributions.gaussian as gau

if (__name__ == "__main__"):
    XGAU = np.load('../Data/XGau.npy')

    print("XGAU DATA:")
    print(XGAU)
    print("XGAU shape: ", XGAU.shape)
    print("XGAU mean: ", XGAU.mean())
    print("XGAU variance: ", XGAU.var())
    plt.figure()
    plt.title("XGAU Data normalized in 50 bins")
    plt.hist(XGAU, bins=50, density=True)
    print("------\n")

    # Density
    print(
        "Example 1: calculating the gaussian density values corresponding to 1000 equally spaced points between -8 and 12..")
    plt.figure()
    plt.title("Example 1 - gau values")
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot, gau.GAU_pdf(XPlot, 1.0, 2.0))
    # Check correctness against solution
    pdfSol = np.load("../Solution/CheckGAUPdf.npy")
    pdfGau = gau.GAU_pdf(XPlot, 1.0, 2.0)
    print("Absolute difference from solution: ", np.abs(pdfSol - pdfGau).mean())
    print("------\n")

    print(
        "Example 2: calculating the gaussian density values of the XGAU data assuming a gaussian distribution with mean 0 and variance 1..")
    ll_samples = gau.GAU_pdf(XGAU, 0.0, 1.0)
    print("Probability Density Function values: ")
    print(ll_samples)
    likelihood = ll_samples.prod()
    print("Likelihood: ", likelihood)
    print("The values are very small and the product of these values saturates to zero!")
    print("------\n")

    print(
        "Example 3: calculating the logarithm gaussian density values of the XGAU data assuming a gaussian distribution with mean 0 and variance 1..")
    ll_log_samples = gau.GAU_logpdf(XGAU, 0.0, 1.0)
    print("Logarithm Probability Density Function values: ")
    print(ll_log_samples)
    log_likelihood = ll_log_samples.sum()
    print("Logarithm Likelihood: ", log_likelihood)
    print("------\n")

    # Maximum logarithm likelihood
    print("Estimating the gaussian parameters using the Maximum Likelihood estimator technique..")
    mu_ML = XGAU.mean()
    var_ML = XGAU.var()
    ll_log_samples = gau.GAU_logpdf(XGAU, mu_ML, var_ML)
    print("Logarithm Probability Density Function values: ")
    print(ll_log_samples)
    ll = ll_log_samples.sum()
    print("ML estimated: ", ll)
    plt.figure()
    plt.title(
        "Real XGAU histogram data against 1000 points from -8 to 12 of a gaussian using the estimated mean and variance")
    plt.hist(XGAU, bins=50, density=True)
    plt.plot(XPlot, np.exp(gau.GAU_logpdf(XPlot, mu_ML, var_ML)))
    print("------\n")

    # Maximum log likelihood with too little samples
    counts = [1000, 500, 100, 50, 10]
    for i in counts:
        print("Estimating the gaussian parameters using ML with %d samples.." % i)
        mu_ML = XGAU[0:i].mean()
        var_ML = XGAU[0:i].var()
        ll_log_samples = gau.GAU_logpdf(XGAU, mu_ML, var_ML)
        ll = ll_log_samples.sum()
        print("ML estimated: ", ll)
        plt.figure()
        plt.title("Gaussian estimated with %d samples" % i)
        plt.hist(XGAU, bins=50, density=True)
        plt.plot(XPlot, np.exp(gau.GAU_logpdf(XPlot, mu_ML, var_ML)))

    print("\nWe can see that with too little samples, the curve doesn't fit very well the data anymore!")
    print("------\n")

    # Multivariate Gaussian
    print("Multivariate Gaussian example..")
    XND = np.load("../Solution/XND.npy")
    print("XND shape: ", XND.shape)
    # print(XND)
    mu = np.load("../Solution/muND.npy")
    print("mu shape: ", mu.shape)
    C = np.load("../Solution/CND.npy")
    print("C shape: ", C.shape)
    pdfSol = np.load("../Solution/llND.npy")
    pdfGau = gau.logpdf_GAU_ND(XND, mu, C)
    print("pdf GAU ND shape: ", pdfGau.shape)
    # print(pdfGau)
    # print(pdfSol)

    print("Difference from computation and solution: ", np.abs(pdfSol - pdfGau).mean())

    plt.show()
