import numpy as np
import scipy.optimize
import scipy.special

class SVM_Classifier:
    def __init__(self):
        self.DTR = None
        self.LTR = None
        self.C = 1
        self.K = 1
        self.kernel = None
        self.w_hat = None
        self.alfas = None

    def train(self, DTR, LTR, C, K, pi1=None, kernel=None, maxfun=15000, maxiter=15000, factr=10000000.0, verbose=0):
        self.DTR = DTR
        self.LTR = LTR
        self.C = C
        self.K = K
        self.pi1 = pi1
        self.kernel = kernel

        if kernel is None:
            self.w_hat, self.alfas = svm_train(DTR, LTR, C, K, pi1, kernel, maxfun, maxiter, factr, verbose)
        else:
            self.alfas = svm_train(DTR, LTR, C, K, pi1, kernel, maxfun, maxiter, factr, verbose)

    def compute_scores(self, D):
        S = svm_compute_scores(D, self.K, self.kernel, self.w_hat, self.alfas, self.DTR, self.LTR)
        return S

    def inference(self, D):
        pred_labels = svm_inference(D, self.K, self.kernel, self.w_hat, self.alfas, self.DTR, self.LTR)
        return pred_labels

    def duality_gap(self, DTR, LTR):
        return svm_duality_gap(DTR, LTR, self.C, self.K, self.w_hat, self.alfas)

    @staticmethod
    def Kernel_Polynomial(d, c):
        def kernel(x1, x2):
            return (x1.T @ x2 + c)**d
        return kernel

    @staticmethod
    def Kernel_RadialBasisFunction(gamma):
        def kernel(x1, x2):
            return np.exp(-gamma * ((x1-x2) * (x1-x2)).sum())
        return kernel


def linear_svm_dual_obj_wrapper(DTR, LTR):
    LTR = LTR.reshape((1, LTR.shape[0]))
    Z = LTR.T @ LTR
    def svm_dual_obj(alfa):
        alfa = alfa.reshape(alfa.shape[0], 1)
        G = DTR.T @ DTR
        H = Z * G
        ones = np.ones(DTR.shape[1]).reshape(DTR.shape[1], 1)
        L_hat_dual = 0.5 * (alfa.T @ H @ alfa) - alfa.T @ ones
        grad = H @ alfa - ones
        grad = grad.reshape(DTR.shape[1])
        return (L_hat_dual, grad)

    return svm_dual_obj

def linear_svm_primal_obj_wrapper(DTR, LTR, C):
    def svm_primal_obj(w):
        w = w.reshape(w.shape[0], 1)
        regularization_term = 0.5 * (w*w).sum()

        tmp = 0.0
        for i in range(DTR.shape[1]):
            xi = DTR[:, i].reshape((DTR.shape[0], 1))
            tmp = tmp + max(0, 1 - LTR[i] * (w.T @ xi))

        tmp = tmp * C
        Jhat = tmp + regularization_term
        return Jhat

    return svm_primal_obj

def svm_dual_obj_wrapper(DTR, LTR, K, kernel):
    def svm_dual_obj(alfa):
        alfa = alfa.reshape(alfa.shape[0], 1)

        # Calculate H
        H = np.zeros((DTR.shape[1], DTR.shape[1]))
        for i in range(DTR.shape[1]):
            xi = DTR[:, i].reshape(DTR.shape[0], 1)
            for j in range(DTR.shape[1]):
                if (j >= i):
                    xj = DTR[:, j].reshape(DTR.shape[0], 1)
                    H[i, j] = LTR[i] * LTR[j] * (kernel(xi, xj) + K**2)
                    if i != j:
                        H[j, i] = H[i, j]

        ones = np.ones(DTR.shape[1]).reshape(DTR.shape[1], 1)
        # This corresponds to minus the objective function to maximize (J)
        L_hat_dual = 0.5 * (alfa.T @ H @ alfa) - alfa.T @ ones
        grad = H @ alfa - ones
        grad = grad.reshape(DTR.shape[1])
        return (L_hat_dual, grad)

    return svm_dual_obj

def svm_train(DTR, LTR, C, K, pi1=None, kernel=None, maxfun=15000, maxiter=15000, factr=10000000.0, verbose=0):
    if verbose:
        print("Training SVM classifier..")
        print("C: ", C)
        print("K: ", K)
        if kernel is not None:
            print("Kernel: ", kernel.__name__)

    LTRz = np.array([1 if (label == 1) else -1 for label in LTR])
    if kernel is None:
        DTRhat = np.vstack((DTR, np.ones(DTR.shape[1]).reshape(1, DTR.shape[1]) * K))
        dual_obj = linear_svm_dual_obj_wrapper(DTRhat, LTRz)
    else:
        dual_obj = svm_dual_obj_wrapper(DTR, LTRz, K, kernel)

    x0 = np.zeros(DTR.shape[1])

    if pi1 is not None:
        pi1_emp = DTR[:, (LTR==1)].shape[1] / DTR.shape[1]
        C1 = C * pi1 / pi1_emp
        C0 = C * (1 - pi1) / (1 - pi1_emp)
        constraints = [(0, C0) if LTR[i] == 0 else (0, C1) for i in range(DTR.shape[1])]
    else:
        constraints = [(0, C) for i in range(DTR.shape[1])]

    xMin, fMin, d = scipy.optimize.fmin_l_bfgs_b(dual_obj, x0, bounds=constraints, maxfun=maxfun, maxiter=maxiter, factr=factr)
    alfas = xMin
    if verbose:
        print("Dual loss: ", -fMin)
    if kernel is None:
        w_hat = (alfas * LTRz * DTRhat).sum(axis=1)
        return w_hat, alfas

    return alfas

def svm_compute_scores(D, K, kernel=None, w_hat=None, alfas=None, DTR=None, LTR=None):
    if kernel is None:
        w_hat = w_hat.reshape(w_hat.shape[0], 1)
        Dhat = np.vstack((D, np.ones(D.shape[1]).reshape(1, D.shape[1]) * K))
        S = w_hat.T @ Dhat
        S = S.flatten()
    else:
        LTRz = np.array([1 if (label == 1) else -1 for label in LTR])

        S = []
        # for each sample, we compute the score
        for i in range(D.shape[1]):
            xt = D[:, i].reshape((DTR.shape[0], 1))
            score = 0.0
            for j in range(LTRz.shape[0]):
                if alfas[j] == 0:
                    continue
                xi = DTR[:, j].reshape((DTR.shape[0], 1))
                score = score + alfas[j] * LTRz[j] * (kernel(xi, xt) + K**2)
            S.append(score)

    return S

def svm_inference(D, K, kernel=None, w_hat=None, alfas=None, DTR=None, LTR=None):
    S = svm_compute_scores(D, K, kernel, w_hat, alfas, DTR, LTR)
    pred_labels = np.array([1 if score > 0 else 0 for score in S])

    return pred_labels

def svm_duality_gap(DTR, LTR, C, K, w_hat, alfas):
    DTRhat = np.vstack((DTR, np.ones(DTR.shape[1]).reshape(1, DTR.shape[1]) * K))
    LTRz = np.array([1 if (label == 1) else -1 for label in LTR])

    primal_obj = linear_svm_primal_obj_wrapper(DTRhat, LTRz, C)
    dual_obj = linear_svm_dual_obj_wrapper(DTRhat, LTRz)

    primal_loss = primal_obj(w_hat)
    dual_loss = -dual_obj(alfas)[0]

    duality_gap = primal_loss - dual_loss

    return duality_gap, primal_loss, dual_loss
