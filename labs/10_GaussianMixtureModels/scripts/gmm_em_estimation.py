import os
import numpy as np
import matplotlib.pyplot as plt

import classifiers.distributions.gaussian as gau
import density_estimation.gaussian_mixture_model as gmm

import time

SCRIPT_PATH = os.path.dirname(__file__)
GMM_SAMPLE_DATA_4D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_data_4D.npy")
GMM_4D_3G_INITIAL_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_4D_3G_init.json")
GMM_4D_3G_SOLUTION_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_4D_3G_EM.json")


GMM_SAMPLE_DATA_1D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_data_1D.npy")
GMM_1D_3G_INITIAL_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_1D_3G_init.json")
GMM_1D_3G_SOLUTION_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_1D_3G_EM.json")

if __name__ == "__main__":
    # 4D
    print("4D DATASET ANALYSYS..")
    X_4D = np.load(GMM_SAMPLE_DATA_4D_PATH)
    gmm_4D_initial = gau.load_gmm(GMM_4D_3G_INITIAL_PATH)

    time_start = time.perf_counter()
    gmm_em_4D = gmm.gmm_em_estimate(X_4D, gmm_4D_initial, avg_ll_threshold=1e-6, verbose=1)
    time_end = time.perf_counter()
    print("GMM EM ESTIMATION:")
    print(gmm_em_4D)
    print("")
    gmm_4d_solution = gau.load_gmm(GMM_4D_3G_SOLUTION_PATH)
    print("GMM EM SOLUTION:")
    print(gmm_4d_solution)
    print("----------------------------------\n\n")
    print("EM estimation time: %d seconds" % (time_end-time_start))

    # 1D
    print("1D DATASET ANALYSYS..")
    X_1D = np.load(GMM_SAMPLE_DATA_1D_PATH)
    gmm_1D_initial = gau.load_gmm(GMM_1D_3G_INITIAL_PATH)

    time_start = time.perf_counter()
    gmm_em_1D = gmm.gmm_em_estimate(X_1D, gmm_1D_initial, avg_ll_threshold=1e-6, verbose=1)
    time_end = time.perf_counter()
    print("GMM EM ESTIMATION:")
    print(gmm_em_1D)
    print("")
    gmm_1d_solution = gau.load_gmm(GMM_1D_3G_SOLUTION_PATH)
    print("GMM EM SOLUTION:")
    print(gmm_1d_solution)
    print("EM estimation time: %d seconds" % (time_end-time_start))

    # Plotting 1D histogram against estimated GMM density
    plt.figure()
    plt.title("EM ESTIMATED GMM VERSUS ACTUAL DATA HISTOGRAM")
    plt.hist(X_1D.flatten(), density=True, bins=30, histtype='bar', ec='black')
    XPlot = np.linspace(-10, 5, 1000)
    XPlot = XPlot.reshape((1, XPlot.shape[0]))
    y = np.exp(gau.logpdf_GMM(XPlot, gmm_em_1D))
    XPlot = XPlot.flatten()
    plt.plot(XPlot, y)

    plt.show()