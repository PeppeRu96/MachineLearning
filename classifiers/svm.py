import numpy as np
import scipy.optimize
import scipy.special
from scipy.spatial import distance_matrix

import preproc.dstools as dst

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
        def kernel(D1, D2):
            return (D1.T @ D2 + c) ** d
        return kernel

    @staticmethod
    def Kernel_RadialBasisFunction(gamma):
        def kernel(D1, D2):
            return np.exp(-gamma * (distance_matrix(D1.T, D2.T) ** 2))
        return kernel

def svm_dual_obj_wrapper(DTR, LTR, K=None, kernel=None):
    Z = 2 * LTR - 1
    Z = Z.reshape((1, Z.shape[0]))
    Z = Z.T @ Z
    if kernel is None:
        G = DTR.T @ DTR
    else:
        G1 = kernel(DTR, DTR)
        if K is not None:
             G = G1 + K**2

    H = Z * G

    ones = np.ones(DTR.shape[1]).reshape(DTR.shape[1], 1)
    def svm_dual_obj(alfa):
        alfa = alfa.reshape(alfa.shape[0], 1)
        L_hat_dual = 0.5 * (alfa.T @ H @ alfa) - alfa.T @ ones
        grad = (H @ alfa - ones).flatten()
        return (L_hat_dual, grad)

    return svm_dual_obj

def linear_svm_primal_obj_wrapper(DTR, LTR, C):
    Z = LTR * 2 - 1
    def svm_primal_obj(w):
        w = w.reshape(w.shape[0], 1)
        regularization_term = 0.5 * (w*w).sum()

        tmp = 0.0
        for i in range(DTR.shape[1]):
            xi = DTR[:, i].reshape((DTR.shape[0], 1))
            tmp = tmp + max(0, 1 - Z[i] * (w.T @ xi))

        tmp = tmp * C
        Jhat = tmp + regularization_term
        return Jhat

    return svm_primal_obj

def svm_train(DTR, LTR, C, K, pi1=None, kernel=None, maxfun=15000, maxiter=15000, factr=10000000.0, verbose=0):
    if verbose:
        print("Training SVM classifier..")
        print("C: ", C)
        print("K: ", K)
        if kernel is not None:
            print("Kernel: ", kernel.__name__)

    if kernel is None:
        DTRhat = np.vstack((DTR, np.ones(DTR.shape[1]).reshape(1, DTR.shape[1]) * K))
        dual_obj = svm_dual_obj_wrapper(DTRhat, LTR)
    else:
        dual_obj = svm_dual_obj_wrapper(DTR, LTR, K, kernel)

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
        Z = 2 * LTR - 1
        w_hat = (alfas * Z * DTRhat).sum(axis=1)
        w_hat = w_hat.reshape(w_hat.shape[0], 1)
        return w_hat, alfas

    return alfas

def svm_compute_scores(D, K, kernel=None, w_hat=None, alfas=None, DTR=None, LTR=None):
    if kernel is None:
        Dhat = np.vstack((D, np.ones(D.shape[1]).reshape(1, D.shape[1]) * K))
        S = (w_hat.T @ Dhat).flatten()
    else:
        Z = LTR * 2 - 1
        Z = Z.reshape(Z.shape[0], 1)
        alfas = alfas.reshape(alfas.shape[0], 1)
        S = ((kernel(DTR, D) + K ** 2) * Z * alfas).sum(axis=0)

    return S

def svm_inference(D, K, kernel=None, w_hat=None, alfas=None, DTR=None, LTR=None):
    S = svm_compute_scores(D, K, kernel, w_hat, alfas, DTR, LTR)
    pred_labels = np.array([1 if score > 0 else 0 for score in S])

    return pred_labels

def svm_duality_gap(DTR, LTR, C, K, w_hat, alfas):
    DTRhat = np.vstack((DTR, np.ones(DTR.shape[1]).reshape(1, DTR.shape[1]) * K))

    primal_obj = linear_svm_primal_obj_wrapper(DTRhat, LTR, C)
    dual_obj = svm_dual_obj_wrapper(DTRhat, LTR)

    primal_loss = primal_obj(w_hat)
    dual_loss = -dual_obj(alfas)[0]

    duality_gap = primal_loss - dual_loss

    return duality_gap, primal_loss, dual_loss

def cross_validate_svm(preproc_conf, C, K, X_train, y_train, X_test=None, y_test=None, X_val=None, y_val=None, specific_pi1=None, kernel=None, maxfun=15000, maxiter=15000, factr=1.0):
    svm = None
    def train_and_validate(DTR, LTR, DTE, LTE):
        global svm
        # Preprocess data
        DTR, DTE = preproc_conf.apply_preproc_pipeline(DTR, LTR, DTE)

        # Train
        svm = SVM_Classifier()
        svm.train(DTR, LTR, C, K, specific_pi1, kernel, maxfun, maxiter, factr)

        # Validate
        s = svm.compute_scores(DTE)
        return s

    def validate(DTR, LTR, DTV, LTV):
        global svm
        # Preprocess data
        DTR, DTV = preproc_conf.apply_preproc_pipeline(DTR, LTR, DTV)

        # Validate
        s = svm.compute_scores(DTV)
        return s

    if X_test is None:
        # Cross-validation
        iterations = 1
        scores = []
        labels = []
        for DTR, LTR, DTE, LTE in dst.kfold_generate(X_train, y_train):
            s = train_and_validate(DTR, LTR, DTE, LTE)

            # Collect scores and associated labels
            scores.append(s)
            labels.append(LTE)

            iterations += 1

        scores = np.array(scores).flatten()
        labels = np.array(labels).flatten()
        scores_val = None
        labels_val = None
    else:
        # Standard train-validation on fixed split
        scores = train_and_validate(X_train, y_train, X_test, y_test)
        scores = scores.flatten()
        labels = y_test
        if X_val is not None:
            scores_val = validate(X_train, y_train, X_val, y_val)
            scores_val = scores_val.flatten()
            labels_val = y_val
        else:
            scores_val = None
            labels_val = None

    return scores, labels, scores_val, labels_val
