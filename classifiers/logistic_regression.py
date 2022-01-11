import numpy as np
import scipy.optimize
import scipy.special

import preproc.dstools as dst

class LogisticRegressionClassifier:
    def __init__(self, K):
        self.K = K
        self.DTR = None
        self.LTR = None
        self.l = 0
        self.expand_feature_space_func = None
        self.w = None
        self.b = None

    def train(self, DTR, LTR, l, pi1=None, expand_feature_space_func=None, maxfun=15000, maxiter=15000, factr=1e7,
              verbose=0):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.pi1 = pi1
        self.expand_feature_space_func = expand_feature_space_func

        self.w, self.b = LR_Classifier_train(DTR, LTR, self.K, l, pi1, expand_feature_space_func, maxfun, maxiter, factr,
                                             verbose)

    def inference(self, D):
        return LR_Classifier_inference(D, self.w, self.b, self.expand_feature_space_func)

    def compute_binary_classifier_llr(self, D):
        return LR_Classifier_compute_llr(D, self.w, self.b, self.expand_feature_space_func)

    @staticmethod
    def quadratic_feature_expansion(x):
        """
        Computes the quadratic expansion phi(x) = [vec(x@x.T), x].T of one or more column vectors
        :param x: feature vector (it has to be a column vector) or a matrix of column feature vectors
        :return: a column vector containing the expanded input vector or a matrix of column expanded vectors
        """
        if (x.shape[1] > 1):
            X_t = x.T
            X = X_t.reshape(X_t.shape[0], X_t.shape[1], 1)
            phi_x = X @ X.transpose((0, 2, 1))
            phi_x = phi_x.reshape((phi_x.shape[0], phi_x.shape[1] * phi_x.shape[2]))
            phi_x = phi_x.T
            phi_x = np.vstack((phi_x, x))
        else:
            m = x @ x.T
            phi_x = m.flatten(order='F')
            phi_x = phi_x.reshape((phi_x.shape[0], 1))
            phi_x = np.vstack((phi_x, x))

        return phi_x

# Optimized for efficiency (it requires some additional memory due to additional views of the same data)
def logreg_obj_wrapper(DTR, LTR, l, pi1=None):
    def compute_sum(D, Z, w, b):
        tmp = - Z * ( (w.T @ D).flatten() + b )
        return np.logaddexp(np.array([0]), tmp).sum()

    def compute_sum0(D, w, b):
        tmp = (w.T @ D).flatten() + b
        return np.logaddexp(np.array([0]), tmp).sum()

    def compute_sum1(D, w, b):
        tmp = -1 * ( (w.T @ D).flatten() + b )
        return np.logaddexp(np.array([0]), tmp).sum()

    regularizer = 0.5 * l
    Z = 2 * LTR - 1
    DTR0 = DTR[:, (LTR == 0)]
    DTR1 = DTR[:, (LTR == 1)]

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        w = w.reshape((w.shape[0], 1))
        regularization_term = regularizer * (w ** 2).sum()

        J = compute_sum(DTR, Z, w, b) / DTR.shape[1] + regularization_term
        return J

    def logreg_obj_pi1(v):
        w, b = v[0:-1], v[-1]
        w = w.reshape((w.shape[0], 1))
        regularization_term = regularizer * (w ** 2).sum()

        J0 = compute_sum0(DTR0, w, b) * (1 - pi1) / DTR0.shape[1]
        J1 = compute_sum1(DTR1, w, b) * pi1 / DTR1.shape[1]
        J = J0 + J1 + regularization_term
        return J

    if pi1 is None:
        return logreg_obj

    return logreg_obj_pi1


def LR_Classifier_train(DTR, LTR, K, l, pi1=None, expand_feature_space_func=None, maxfun=15000, maxiter=15000, factr=1e7,
                        verbose=0):
    # K: number of classes
    if verbose:
        print("Training Logistic Regression classifier using Average Risk Minimizer method..")
        print("Lambda regularizer: ", l)

    if expand_feature_space_func is not None:
        DTR = expand_feature_space_func(DTR)

    D = DTR.shape[0]

    # Multiclass or binary LR
    if K > 2:
        logreg_obj = logreg_multiclass_obj_wrapper(DTR, LTR, K, l)
        x0 = np.zeros(D * K + K)
    else:
        logreg_obj = logreg_obj_wrapper(DTR, LTR, l, pi1=pi1)
        x0 = np.zeros(D + 1)

    xMin, fMin, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, maxfun=maxfun, maxiter=maxiter, factr=factr,
                                                 approx_grad=True)

    if K > 2:
        w, b = xMin[0:D * K].reshape((D, K)), xMin[D * K:].reshape((K, 1))
    else:
        w, b = xMin[0:-1], xMin[-1]
        w = w.reshape((w.shape[0], 1))

    if verbose:
        print("J(w*, b*): ", fMin)

    return w, b


def LR_Classifier_compute_llr(D, w, b, expand_feature_space_func=None):
    if expand_feature_space_func is not None:
        D = expand_feature_space_func(D)

    S = (w.T @ D + b).flatten()

    return S


def LR_Classifier_inference(D, w, b, expand_feature_space_func=None):
    if expand_feature_space_func is not None:
        D = expand_feature_space_func(D)

    S = (w.T @ D + b)

    if w.shape[1] > 1:
        pred_labels = np.argmax(S, 0)
    else:
        pred_labels = np.array([1 if score > 0 else 0 for score in S.flatten()])
    return pred_labels


# MULTICLASS LR
def logreg_multiclass_obj_wrapper(DTR, LTR, K, l):
    def logreg_multiclass_obj(v):
        D = DTR.shape[0]  # dimensionality of feature

        W = v[0:D * K].reshape((D, K))
        b = v[D * K:].reshape((K, 1))
        regularization_term = 0.5 * l * (W * W).sum()

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


def cross_validate_lr(preproc_conf, lambda_regularizer, X_train, y_train, X_test=None, y_test=None, pi1=None, quadratic=False):
    pi1_str = "with prior weight specific training (π=%.1f)" % (pi1) if pi1 is not None else ""

    def train_and_validate(DTR, LTR, DTE, LTE):
        # Preprocess data
        DTR, DTE = preproc_conf.apply_preproc_pipeline(DTR, LTR, DTE)

        # Train
        lr = LogisticRegressionClassifier(2)
        efs = LogisticRegressionClassifier.quadratic_feature_expansion if quadratic else None
        lr.train(DTR, LTR, lambda_regularizer, pi1=pi1, expand_feature_space_func=efs)

        # Validate
        s = lr.compute_binary_classifier_llr(DTE)
        return s

    if X_test is None:
        # Cross-validation
        print("\t\t5-Fold Cross-Validation %s LR %s (λ=%.5f) - Preprocessing: %s" %
              ("Quadratic" if quadratic else "Linear", pi1_str, lambda_regularizer, preproc_conf))
        iterations = 1
        scores = []
        labels = []
        for DTR, LTR, DTE, LTE in dst.kfold_generate(X_train, y_train):
            # Preprocess data
            DTR, DTE = preproc_conf.apply_preproc_pipeline(DTR, LTR, DTE)

            # Train
            lr = LogisticRegressionClassifier(2)
            efs = LogisticRegressionClassifier.quadratic_feature_expansion if quadratic else None
            lr.train(DTR, LTR, lambda_regularizer, pi1=pi1, expand_feature_space_func=efs)

            # Validate
            s = train_and_validate(DTR, LTR, DTE, LTE)

            # Collect scores and associated labels
            scores.append(s)
            labels.append(LTE)

            iterations += 1

        scores = np.array(scores).flatten()
        labels = np.array(labels).flatten()
    else:
        # Standard train-validation on fixed split
        print("\t\tTrain and validation %s LR %s (λ=%.5f) - Preprocessing: %s" %
              ("Quadratic" if quadratic else "Linear", pi1_str, lambda_regularizer, preproc_conf))
        scores = train_and_validate(X_train, y_train, X_test, y_test)
        scores = scores.flatten()
        labels = y_test

    return scores, labels
