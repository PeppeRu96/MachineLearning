import numpy as np
import scipy.optimize
import scipy.special

class LogisticRegressionClassifier:
    def __init__(self):
        self.DTR = None
        self.LTR = None
        self.l = 0
        self.w = None
        self.b = None

    def train(self, DTR, LTR, l, maxfun=15000, maxiter=15000, verbose=0):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l

        self.w, self.b = LR_Classifier_train(DTR, LTR, l, maxfun, maxiter, verbose)

    def inference(self, D):
        pred_labels = LR_Classifier_inference(D, self.w, self.b)
        return pred_labels

def logreg_obj_wrapper(DTR, LTR, l):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        regularization_term = 0.5 * l * (w**2).sum()

        J = 0
        for i in range(DTR.shape[1]):
            xi = DTR[:, i]
            ci = LTR[i]
            tmp = (w.T @ xi) + b
            J = J + ci * np.log1p(np.exp(-tmp)) + (1 - ci) * np.log1p(np.exp(tmp))

        J = J / DTR.shape[1]
        J = J + regularization_term
        return J

    return logreg_obj

def LR_Classifier_train(DTR, LTR, l, maxfun=15000, maxiter=15000, verbose=0):
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
        logreg_obj = logreg_obj_wrapper(DTR, LTR, l)
        x0 = np.zeros(D + 1)

    xMin, fMin, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, maxfun=maxfun, maxiter=maxiter, approx_grad=True)

    if K > 2:
        w, b = xMin[0:D*K].reshape((D, K)), xMin[D*K:].reshape((K, 1))
    else:
        w, b = xMin[0:-1].T, xMin[-1]
    if verbose:
        #print("Logistic Regression W direction and bias b:")
        #print("w*: ", w)
        #print("b*: ", b)
        print("J(w*, b*): ", fMin)

    return w, b

def LR_Classifier_inference(D, w, b):
    K = w.shape[1]
    S = w.T @ D + b

    if K > 2:
        pred_labels = np.argmax(S, 0)
    else:
        pred_labels = np.array([1 if score > 0 else 0 for score in S])
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