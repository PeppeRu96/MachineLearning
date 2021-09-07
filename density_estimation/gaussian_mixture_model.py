import numpy as np
import scipy.special

import classifiers.distributions.gaussian as gau

def gmm_em_estimate(X, gmm0, psi=0.01, diag_cov=False, tied_cov=False, avg_ll_threshold=1e-6, verbose=0):
    if verbose:
        print("Estimating GMM through Expectation Maximization algorithm..")

    curr_gmm = gmm0
    previous_avg_log_likelihood = None
    iterations = 0
    while(1):
        G = len(curr_gmm)
        # E-step: compute responsibilities from the previous gmm estimate
        S_joint_log_density = []
        for (w, mu, sigma) in curr_gmm:
            joint_log_density_g = gau.logpdf_GAU_ND(X, mu, sigma) + np.log(w)
            S_joint_log_density.append(joint_log_density_g)
        S_joint_log_density = np.array(S_joint_log_density)
        gmm_log_density = scipy.special.logsumexp(S_joint_log_density, axis=0)

        # Stopping criterion
        current_avg_log_likelihood = gmm_log_density.sum() / X.shape[1]
        if previous_avg_log_likelihood is not None and current_avg_log_likelihood - previous_avg_log_likelihood < avg_ll_threshold:
            break;
        previous_avg_log_likelihood = current_avg_log_likelihood

        S_posterior_prob = np.exp(S_joint_log_density - gmm_log_density)

        # M-step: update the model parameters using the just estimated responsibilities
        # Calculating statistics
        Sprime = S_posterior_prob.reshape(G, 1, X.shape[1])
        Z = S_posterior_prob.sum(axis=1)
        F = (Sprime * X).sum(axis=2)
        S = (Sprime * X) @ X.T
        Zsum = Z.sum()

        # Calculating new gmm parameters from statistics
        W = Z / Zsum
        Z1 = Z.reshape(Z.shape[0], 1)
        M = F / Z1
        M = M.reshape(M.shape[0], M.shape[1], 1)
        Z1 = Z.reshape(Z.shape[0], 1, 1)
        Sigma = S / Z1 - M @ M.transpose((0, 2, 1))
        # Diagonal and tied covariances
        if diag_cov:
            Sigma = Sigma * np.eye(Sigma.shape[1])
        if tied_cov:
            Stied = (Sigma * Z1).sum(axis=0) / X.shape[1]
            Sigma[:] = Stied
        # Constraining GMM
        U, s, _ = np.linalg.svd(Sigma)
        s[s < psi] = psi
        s = s.reshape((s.shape[0], s.shape[1], 1))
        Sigma = U @ (s * U.transpose(0, 2, 1))

        curr_gmm = [(w, mu, sigma) for w, mu, sigma in zip(W, M, Sigma)]
        iterations += 1

    if verbose:
        print("EM estimation finished..")

    return curr_gmm

def GMM_1_ML(X):
    # Starting GMM with 1 component: ML of the MVG
    w_1 = 1.0
    mu_1 = X.mean(1)
    mu_1 = mu_1.reshape((mu_1.shape[0], 1))
    tmp = X - mu_1
    sigma_0 = (tmp @ tmp.T) / X.shape[1]
    GMM_1 = [(w_1, mu_1, sigma_0)]

    return GMM_1

def LBG_estimate(X, alpha, psi=0.01, diag_cov=False, tied_cov=False, gmm_init=None, step_fun=None, stop_condition_fun=None, verbose=0):
    if gmm_init is None:
        gmm_init = GMM_1_ML(X)

    if step_fun is None:
        step_fun = lambda x: 2*x

    if stop_condition_fun is None:
        stop_condition_fun = lambda gmm: len(gmm) >= len(gmm_init) * 2

    if diag_cov:
        gmm_init = gmm_covariance_diagonal(gmm_init)

    if tied_cov:
        gmm_init = gmm_covariance_tied(gmm_init, X)

    # Avoid degenerate solutions
    gmm_init = gmm_covariance_eigenvalues_constrained(gmm_init, psi)

    if verbose:
        print("Estimating GMM through LBG algorithm..")
        print("Starting GMM component count: ", len(gmm_init))
        print("Step function: %d -> %d" % (len(gmm_init), step_fun(len(gmm_init))))

    all_gmm = []
    all_gmm.append(gmm_init)

    curr_gmm = gmm_init
    iterations = 1
    while not stop_condition_fun(curr_gmm):
        if verbose:
            print("Iteration ", iterations)
        # Splitting current GMM in step_fun(len(current gmm))
        new_components_cnt = step_fun(len(curr_gmm))
        new_components_len = len(curr_gmm)
        new_gmm = []
        for component in curr_gmm:
            w_g, mu_g, sigma_g = component[0], component[1], component[2]
            U, s, Vh = np.linalg.svd(sigma_g)
            d_g = U[:, 0:1] * s[0] ** 0.5 * alpha

            if new_components_len < new_components_cnt:
                component_1 = (w_g/2, mu_g - d_g, sigma_g)
                component_2 = (w_g/2, mu_g + d_g, sigma_g)
                new_gmm.append(component_1)
                new_gmm.append(component_2)
                new_components_len += 1
            else:
                new_gmm.append(component)
        if verbose:
            print("New gmm component count: ", len(new_gmm))
        new_gmm = gmm_em_estimate(X, new_gmm, psi=psi, diag_cov=diag_cov, tied_cov=tied_cov)
        curr_gmm = new_gmm
        all_gmm.append(curr_gmm)
        iterations += 1

    return all_gmm

def gmm_covariance_diagonal(gmm):
    gmm_diagonal = []
    for c in gmm:
        w, mu, sigma = c[0], c[1], c[2]
        sigma = sigma * np.eye(sigma.shape[0])

        gmm_diagonal.append((w, mu, sigma))

    return gmm_diagonal

def gmm_covariance_tied(gmm, X):
    S_joint_log_density = []
    for gmm_component in gmm:
        w_g, mu_g, sigma_g = gmm_component[0], gmm_component[1], gmm_component[2]
        log_class_conditional_density_g = gau.logpdf_GAU_ND(X, mu_g, sigma_g)
        joint_log_density_g = log_class_conditional_density_g + np.log(w_g)
        S_joint_log_density.append(joint_log_density_g)
    S_joint_log_density = np.array(S_joint_log_density)
    gmm_log_density = scipy.special.logsumexp(S_joint_log_density, axis=0)
    S_log_posterior_prob = S_joint_log_density - gmm_log_density
    S_posterior_prob = np.exp(S_log_posterior_prob)
    Z = S_posterior_prob.sum(axis=1)

    dim = gmm[0][2].shape[0]
    tied_cov_matr = np.zeros((dim, dim))
    for g in range(len(gmm)):
        sigma_g = gmm[g][2]
        Z_g = Z[g]
        tied_cov_matr += Z_g * sigma_g
    tied_cov_matr = tied_cov_matr / X.shape[1]
    tied_gmm = []
    for component in gmm:
        w_g, mu_g, sigma_g = component[0], component[1], component[2]
        sigma_g = tied_cov_matr
        tied_gmm.append((w_g, mu_g, sigma_g))
    gmm = tied_gmm

    return gmm

def gmm_covariance_eigenvalues_constrained(gmm, psi):
    # Avoid degenerate solutions
    gmm_constrained = []
    for c in gmm:
        w, mu, sigma = c[0], c[1], c[2]
        U, s, _ = np.linalg.svd(sigma)
        s[s < psi] = psi
        s = s.reshape(s.shape[0], 1)
        sigma = U @ (s * U.T)

        gmm_constrained.append((w, mu, sigma))

    return gmm_constrained
