import os
import numpy as np
import matplotlib.pyplot as plt

import classifiers.distributions.gaussian as gau
import density_estimation.gaussian_mixture_model as gmm

SCRIPT_PATH = os.path.dirname(__file__)
GMM_SAMPLE_DATA_4D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_data_4D.npy")
GMM_4D_4G_SOLUTION_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_4D_4G_EM_LBG.json")


GMM_SAMPLE_DATA_1D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_data_1D.npy")
GMM_1D_4G_SOLUTION_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_1D_4G_EM_LBG.json")

if __name__ == "__main__":
    # 4D
    print("4D DATASET ANALYSYS..")
    X_4D = np.load(GMM_SAMPLE_DATA_4D_PATH)

    gmm_em_lbg_4D_all = gmm.LBG_estimate(X_4D, 0.1, stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == 4), verbose=0)
    gmm_em_lbg_4D = gmm_em_lbg_4D_all[-1]
    print("GMM EM LBG ESTIMATION:")
    print(gmm_em_lbg_4D)
    print("")
    gmm_4d_solution = gau.load_gmm(GMM_4D_4G_SOLUTION_PATH)
    print("GMM EM LBG SOLUTION:")
    print(gmm_4d_solution)
    print("----------------------------------\n\n")

    # 1D
    print("1D DATASET ANALYSYS..")
    X_1D = np.load(GMM_SAMPLE_DATA_1D_PATH)

    gmm_em_lbg_1D_all = gmm.LBG_estimate(X_1D, 0.1, stop_condition_fun=(lambda curr_gmm: len(curr_gmm) == 4), verbose=0)
    gmm_em_lbg_1D = gmm_em_lbg_1D_all[-1]
    print("GMM EM LBGESTIMATION:")
    print(gmm_em_lbg_1D)
    print("")

    avg_ll = gau.logpdf_GMM(X_1D, gmm_em_lbg_1D).sum() / X_1D.shape[1]
    print("GMM final average log-likelihood: ", avg_ll)

    gmm_1d_solution = gau.load_gmm(GMM_1D_4G_SOLUTION_PATH)
    print("GMM EM LBG SOLUTION:")
    print(gmm_1d_solution)

    # Plotting 1D histogram against estimated GMM density
    plt.figure()
    plt.title("EM LBG ESTIMATED GMM VERSUS ACTUAL DATA HISTOGRAM")
    plt.hist(X_1D.flatten(), density=True, bins=30, histtype='bar', ec='black')
    XPlot = np.linspace(-10, 5, 1000)
    XPlot = XPlot.reshape((1, XPlot.shape[0]))
    y = np.exp(gau.logpdf_GMM(XPlot, gmm_em_lbg_1D))
    XPlot = XPlot.flatten()
    plt.plot(XPlot, y)

    plt.show()