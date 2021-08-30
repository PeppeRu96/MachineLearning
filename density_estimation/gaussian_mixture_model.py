import numpy as np
import scipy.special

import classifiers.distributions.gaussian as gau

def gmm_em_estimate(X, gmm0, psi=0.01, diag_cov=False, tied_cov=False, avg_ll_threshold=1e-6, verbose=0):
    if verbose:
        print("Estimating GMM through Expectation Maximization algorithm..")

    curr_gmm = gmm0
    previous_avg_log_likelihood = gau.logpdf_GMM(X, curr_gmm).sum() / X.shape[1]
    iterations = 0
    while(1):
        S_joint_log_density = []
        for gmm_component in curr_gmm:
            w_g, mu_g, sigma_g = gmm_component[0], gmm_component[1], gmm_component[2]
            log_class_conditional_density_g = gau.logpdf_GAU_ND(X, mu_g, sigma_g)
            joint_log_density_g = log_class_conditional_density_g + np.log(w_g)
            S_joint_log_density.append(joint_log_density_g)
        S_joint_log_density = np.array(S_joint_log_density)
        gmm_log_density = scipy.special.logsumexp(S_joint_log_density, axis=0)
        S_log_posterior_prob = S_joint_log_density - gmm_log_density
        S_posterior_prob = np.exp(S_log_posterior_prob)

        # Estimating new gmm
        new_gmm = []

        # Calculating statistics
        Z = []
        F = []
        S = []
        for g in range(len(curr_gmm)):
            Z_g = S_posterior_prob[g, :].sum()
            F_g = (S_posterior_prob[g, :] * X).sum(axis=1)
            S_g = (S_posterior_prob[g, :] * X) @ X.T

            Z.append(Z_g)
            F.append(F_g)
            S.append(S_g)

        Zsum = np.array(Z).sum()

        # Calculating new gmm parameters from statistics
        for g in range(len(curr_gmm)):
            Z_g = Z[g]
            F_g = F[g]
            S_g = S[g]

            mu_g = F_g / Z_g
            mu_g = mu_g.reshape(mu_g.shape[0], 1)

            sigma_g = S_g / Z_g - mu_g @ mu_g.T
            if diag_cov:
                sigma_g = sigma_g * np.eye(sigma_g.shape[0])

            w_g = Z_g / Zsum

            new_gmm.append((w_g, mu_g, sigma_g))

        # Tied covariance
        if tied_cov:
            dim = curr_gmm[0][2].shape[0]
            tied_cov_matr = np.zeros((dim, dim))
            for g in range(len(new_gmm)):
                sigma_g = new_gmm[g][2]
                Z_g = Z[g]
                tied_cov_matr += Z_g * sigma_g

            tied_cov_matr = tied_cov_matr / X.shape[1]
            tied_gmm = []
            for component in new_gmm:
                w_g, mu_g, sigma_g = component[0], component[1], component[2]
                sigma_g = tied_cov_matr
                tied_gmm.append((w_g, mu_g, sigma_g))
            new_gmm = tied_gmm

        # Constraint covariance eigenvalues
        new_gmm = gmm_covariance_eigenvalues_constrained(new_gmm, psi)

        # Calculating new log-likelihood on the training set with the estimated parameters
        new_log_likelihood = gau.logpdf_GMM(X, new_gmm).sum()

        # Use the average log-likelihood as a stopping criterion for the EM algorithm
        avg_log_likelihood = new_log_likelihood / X.shape[1]
        increasing = avg_log_likelihood - previous_avg_log_likelihood
        curr_gmm = new_gmm

        if avg_log_likelihood < previous_avg_log_likelihood:
            print("Attention: found an average log likelihood smaller than the previous one!")

        iterations += 1
        if increasing < avg_ll_threshold:
            break

        previous_avg_log_likelihood = avg_log_likelihood

    if verbose:
        print("EM estimation finished..")
        print("Final average log-likelihood on the training set: ", avg_log_likelihood)

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

    # Calculating statistics
    Z = []
    F = []
    S = []
    for g in range(len(gmm)):
        Z_g = S_posterior_prob[g, :].sum()
        F_g = (S_posterior_prob[g, :] * X).sum(axis=1)
        S_g = (S_posterior_prob[g, :] * X) @ X.T

        Z.append(Z_g)
        F.append(F_g)
        S.append(S_g)

    Zsum = np.array(Z).sum()

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
        s = s * np.eye(s.shape[0])
        sigma = U @ s @ U.T

        gmm_constrained.append((w, mu, sigma))

    return gmm_constrained
