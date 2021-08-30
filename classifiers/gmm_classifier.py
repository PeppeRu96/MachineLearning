import numpy as np
import scipy.special

import classifiers.distributions.gaussian as gau

class GMM_Classifier:

    def __init__(self, gmms):
        self.gmms = gmms

    def inference(self, D, Pc):
        S = []
        for i in range(len(self.gmms)):
            log_likelihood_i = gau.logpdf_GMM(D, self.gmms[i])
            lc = log_likelihood_i + np.log(Pc[i])
            S.append(lc)
        Sjoint = np.array(S)
        marginal = scipy.special.logsumexp(Sjoint, axis=0)
        SPost = Sjoint - marginal

        pred_labels = np.argmax(SPost, 0)

        return pred_labels