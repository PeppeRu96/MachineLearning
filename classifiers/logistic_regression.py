import numpy as np
import scipy.optimize
import scipy.special

class LogisticRegressionClassifier:
    def __init__(self):
        self.DTR = None
        self.LTR = None
        self.l = 0
        self.expand_feature_space_func = None
        self.w = None
        self.b = None

    def train(self, DTR, LTR, l, pi1=None, expand_feature_space_func=None, maxfun=15000, maxiter=15000, factr=1e7, verbose=0):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.pi1 = pi1
        self.expand_feature_space_func = expand_feature_space_func

        self.w, self.b = LR_Classifier_train(DTR, LTR, l, pi1, expand_feature_space_func, maxfun, maxiter, factr, verbose)

    def inference(self, D):
        pred_labels = LR_Classifier_inference(D, self.w, self.b, self.expand_feature_space_func)
        return pred_labels

    def compute_binary_classifier_llr(self, D):
        return LR_Classifier_compute_llr(D, self.w, self.b, self.expand_feature_space_func)

    @staticmethod
    def quadratic_feature_expansion(x):
        """
        Computes the quadratic expansion phi(x) = [vec(x@x.T), x].T
        :param x: feature vector (it has to be a column vector)
        :return: a column vector containing the expanded input vector
        """
        m = x @ x.T
        phi_x = m.flatten(order='F')
        phi_x = phi_x.reshape((phi_x.shape[0], 1))
        phi_x = np.vstack((phi_x, x))
        return phi_x


def logreg_obj_wrapper(DTR, LTR, l, pi1=None, expand_feature_space_func=None):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        w = w.reshape((w.shape[0], 1))
        regularization_term = 0.5 * l * (w**2).sum()

        J = 0
        Jv = np.zeros(2)
        for i in range(DTR.shape[1]):
            xi = DTR[:, i]
            xi = xi.reshape(xi.shape[0], 1)
            if expand_feature_space_func is not None:
                xi = expand_feature_space_func(xi)
            ci = LTR[i]
            zi = 2 * ci - 1
            tmp = ((w.T @ xi) + b).flatten()
            if pi1 is None:
                J = J + np.log1p(np.exp(-zi*tmp))
            else:
                Jv[ci] = Jv[ci] + np.log1p(np.exp(-zi*tmp))

        if pi1 is None:
            J = J / DTR.shape[1]
        else:
            Jv[0] = Jv[0] * (1 - pi1) / DTR[:, (LTR == 0)].shape[1]
            Jv[1] = Jv[1] * (pi1) / DTR[:, (LTR == 1)].shape[1]
            J = Jv[0] + Jv[1]

        J = J + regularization_term
        return J

    return logreg_obj

def LR_Classifier_train(DTR, LTR, l, pi1=None, expand_feature_space_func=None, maxfun=15000, maxiter=15000, factr=1e7, verbose=0):
    if verbose:
        print("Training Logistic Regression classifier using Average Risk Minimizer method..")
        print("Lambda regularizer: ", l)
        #print("Max function calls: ", maxfun)
        #print("Max iterations: ", maxiter)

    D = DTR.shape[0]
    K = len(set(LTR))
    # Multiclass or binary LR
    if K > 2:
        logreg_obj = logreg_multiclass_obj_wrapper(DTR, LTR, l)
        x0 = np.zeros(D*K + K)
    else:
        logreg_obj = logreg_obj_wrapper(DTR, LTR, l, pi1=pi1, expand_feature_space_func=expand_feature_space_func)
        if expand_feature_space_func is None:
            x0 = np.zeros(D + 1)
        else:
            x0 = np.zeros(D*(D+1) + 1)

    xMin, fMin, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, maxfun=maxfun, maxiter=maxiter, factr=factr, approx_grad=True)

    if K > 2:
        w, b = xMin[0:D*K].reshape((D, K)), xMin[D*K:].reshape((K, 1))
    else:
        w, b = xMin[0:-1].T, xMin[-1]
        w = w.reshape((w.shape[0], 1))

    if verbose:
        #print("Logistic Regression W direction and bias b:")
        #print("w*: ", w)
        #print("b*: ", b)
        print("J(w*, b*): ", fMin)

    return w, b

def LR_Classifier_compute_llr(D, w, b, expand_feature_space_func=None):
    if expand_feature_space_func is not None:
        dim = D.shape[0]
        Dexpanded = np.zeros((dim*(dim+1), D.shape[1]))
        for i in range(D.shape[1]):
            xi = D[:, i]
            xi = xi.reshape((xi.shape[0], 1))
            xi = expand_feature_space_func(xi)
            Dexpanded[:, i:i+1] = xi
        D = Dexpanded

    S = (w.T @ D + b).flatten()

    return S

def LR_Classifier_inference(D, w, b, expand_feature_space_func=None):
    if expand_feature_space_func is not None:
        dim = D.shape[0]
        Dexpanded = np.zeros((dim * (dim + 1), D.shape[1]))
        for i in range(D.shape[1]):
            xi = D[:, i]
            xi = xi.reshape((xi.shape[0], 1))
            xi = expand_feature_space_func(xi)
            Dexpanded[:, i:i + 1] = xi
        D = Dexpanded

    S = (w.T @ D + b)

    if w.shape[1] > 1:
        pred_labels = np.argmax(S, 0)
    else:
        pred_labels = np.array([1 if score > 0 else 0 for score in S.flatten()])
    return pred_labels

# MULTICLASS LR
def logreg_multiclass_obj_wrapper(DTR, LTR, l):
    def logreg_multiclass_obj(v):
        D = DTR.shape[0]    # dimensionality of feature
        K = len(set(LTR))   # number of classes

        W = v[0:D*K].reshape((D, K))
        b = v[D*K:].reshape((K, 1))
        regularization_term = 0.5 * l * (W*W).sum()

        # Score matrix for all samples
        S = W.T @ DTR + b
        Ylog = S - scipy.special.logsumexp(S, axis=0)

        # 1-of-K encoding for labels
        T = np.zeros((K, LTR.shape[0]))
        for index, label in enumerate(LTR):
            T[label, index] = 1

        J = (T * Ylog).sum()
        J = -J / DTR.shape[1] + regularization_term

        return J

    return logreg_multiclass_obj
