import os
import numpy as np

import classifiers.distributions.gaussian as gau

SCRIPT_PATH = os.path.dirname(__file__)
GMM_SAMPLE_DATA_4D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_data_4D.npy")
GMM_LL_4D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_4D_3G_init_ll.npy")
GMM_DATA_4D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_4D_3G_init.json")

GMM_SAMPLE_DATA_1D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_data_1D.npy")
GMM_LL_1D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_1D_3G_init_ll.npy")
GMM_DATA_1D_PATH = os.path.join(SCRIPT_PATH, "..", "Data", "GMM_1D_3G_init.json")

if __name__ == "__main__":
    # 4D
    gmm_4D = gau.load_gmm(GMM_DATA_4D_PATH)
    X_4D = np.load(GMM_SAMPLE_DATA_4D_PATH)

    gmm_log_density_4D = gau.logpdf_GMM(X_4D, gmm_4D)

    gmm_log_density_solution_4D = np.load(GMM_LL_4D_PATH)

    print("Computed log density GMM 4D:")
    print(gmm_log_density_4D)
    print("")
    print("Solution log density GMM 4D:")
    print(gmm_log_density_solution_4D)

    difference = (gmm_log_density_4D - gmm_log_density_solution_4D).sum()
    print("Sum of difference 4D: ", difference)
    print("----------------\n")



    # 1D
    gmm_1D = gau.load_gmm(GMM_DATA_1D_PATH)
    X_1D = np.load(GMM_SAMPLE_DATA_1D_PATH)

    gmm_log_density_1D = gau.logpdf_GMM(X_1D, gmm_1D)

    gmm_log_density_solution_1D = np.load(GMM_LL_1D_PATH)

    print("Computed log density GMM 1D:")
    print(gmm_log_density_1D)
    print("")
    print("Solution log density GMM 1D:")
    print(gmm_log_density_solution_1D)

    difference = (gmm_log_density_1D - gmm_log_density_solution_1D).sum()
    print("Sum of difference 1D: ", difference)
    print("----------------\n")
