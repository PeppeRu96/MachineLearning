import sklearn.datasets

import preproc.dstools as dst
from classifiers.svm import SVM_Classifier

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


if __name__ == "__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = dst.split_db_2to1(D, L, 2.0 / 3.0)

    Ks = [0.0, 1.0]
    Cs = [1.0]
    kernels = [
        SVM_Classifier.Kernel_Polynomial(2, 0),
        SVM_Classifier.Kernel_Polynomial(2, 1),
        SVM_Classifier.Kernel_RadialBasisFunction(1.0),
        SVM_Classifier.Kernel_RadialBasisFunction(10.0)
    ]
    ker_names = [
        "Poly (d = 2, c = 0)",
        "Poly (d = 2, c = 1)",
        "RBF (gamma = 1.0)",
        "RBF (gamma = 10.0)"
    ]

    print("Trying different combinations of K, C and different kernels on a SVM classifier..")
    print(
        "----------------------------------------------------------------------------------------------------------------")
    print("{:>10}{:>10}{:>30}{:>20}%".format("K", "C", "Kernel", "Error rate"))
    print(
        "----------------------------------------------------------------------------------------------------------------")
    for i, kernel in enumerate(kernels):
        for K in Ks:
            for C in Cs:
                svm = SVM_Classifier()
                svm.train(DTR, LTR, C, K, kernel=kernel, factr=1.0, verbose=0)
                pred_labels = svm.inference(DTE)
                correct_predictions = (pred_labels == LTE).sum()
                accuracy = correct_predictions / pred_labels.shape[0]

                print("{:>10}{:>10.1f}{:>30}{:>20.1f}%".format(K, C, ker_names[i], (1 - accuracy) * 100))
    print(
        "----------------------------------------------------------------------------------------------------------------")
