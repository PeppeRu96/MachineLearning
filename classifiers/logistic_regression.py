import numpy as np
import scipy.optimize

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
        print("Max function calls: ", maxfun)
        print("Max iterations: ", maxiter)

    logreg_obj = logreg_obj_wrapper(DTR, LTR, l)
    x0 = np.zeros(DTR.shape[0] + 1)
    xMin, fMin, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, maxfun=maxfun, maxiter=maxiter, approx_grad=True)

    w, b = xMin[0:-1], xMin[-1]
    if verbose:
        print("Logistic Regression W direction and bias b:")
        print("w*: ", w)
        print("b*: ", b)
        print("J(w*, b*): ", fMin)

    return w.T, b

def LR_Classifier_inference(D, w, b):
    S = w.T @ D + b

    pred_labels = np.array([1 if score > 0 else 0 for score in S])
    return pred_labels

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

