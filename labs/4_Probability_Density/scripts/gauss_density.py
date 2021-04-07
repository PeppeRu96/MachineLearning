import numpy as np
import matplotlib.pyplot as plt

def GAU_pdf(x, mu, var):

    y = np.exp( -((x-mu)**2) / (2*var) ) / np.sqrt(2*np.pi*var)
    print("GAU pdf shape: ", y.shape)
    return y

def GAU_logpdf(x, mu, var):
    y = -0.5*np.log(2*np.pi) - 0.5 * np.log(var) - ((x - mu)**2)/(2*var)
    print("GAU logpdf shape: ", y.shape)
    return y

def logpdf_GAU_ND(x, mu, C):
    M = x.shape(0)
    print("M: ", M)
    for i in range(x.shape(1)):
        ycol = - 0.5 * M * np.log(2*np.pi) - 0.5 * np.linalg.slogdet(C) - 0.5 * (x[:, i] - mu).T * np.linalg.i
    return 0

if (__name__ == "__main__"):
    XGAU = np.load('../Data/XGau.npy')
    print(XGAU)

    print("XGAU shape: ", XGAU.shape)
    print("XGAU mean: ", XGAU.mean())
    print("XGAU variance: ", XGAU.var())
    #plt.figure()
    #plt.hist(XGAU, bins=50, density=True)

    # Density
    #plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    #plt.plot(XPlot, GAU_pdf(XPlot, 1.0, 2.0))
    # Check correctness against solution
    pdfSol = np.load("../Solution/CheckGAUPdf.npy")
    pdfGau = GAU_pdf(XPlot, 1.0, 2.0)
    print("Absolute difference from solution: ", np.abs(pdfSol - pdfGau).mean())

    #ll_samples = GAU_pdf(XGAU, 0.0, 1.0)
    #print("Probability Density Function values: ")
    #print(ll_samples)
    #likelihood = ll_samples.prod()
    #print("Likelihood: ", likelihood)

    #ll_log_samples = GAU_logpdf(XGAU, 0.0, 1.0)
    #print("Logarithm Probability Density Function values: ")
    #print(ll_log_samples)
    #log_likelihood = ll_log_samples.sum()
    #print("Logarithm Likelihood: ", log_likelihood)

    # Maximum logarithm likelihood
    mu_ML = XGAU.mean()
    var_ML = XGAU.var()
    ll_log_samples = GAU_logpdf(XGAU, mu_ML, var_ML)
    print("Logarithm Probability Density Function values: ")
    print(ll_log_samples)
    ll = ll_log_samples.sum()
    print("ML estimated: ", ll)
    plt.figure()
    plt.hist(XGAU, bins=50, density=True)
    plt.plot(XPlot, np.exp(GAU_logpdf(XPlot, mu_ML, var_ML)))

    plt.show()

    # Multivariate Gaussian
    XND = np.load("../Solution/XND.npy")
    print("XND shape: ", XND.shape)
    mu = np.load("../Solution/muND.npy")
    print("mu shape: ", mu.shape)
    C = np.load("../Solution/CND.npy")
    print("C shape: ", C.shape)
    pdfSol = np.load("../Solution/llND.npy")
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print (np.abs(pdfSol - pdfGau).mean())


