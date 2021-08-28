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
    (DTR, LTR), (DTE, LTE) = dst.split_db_2to1(D, L, 2.0/3.0)

    Ks = [1, 10]
    Cs = [0.1, 1.0, 10.0]

    print("Trying different combinations of K and C on a SVM linear classifier..")
    print("----------------------------------------------------------------------------------------------------------------")
    print("{:>10}{:>10}{:>20}{:>20}{:>20}{:>20}%".format("K", "C", "Primal loss", "Dual loss", "Duality gap", "Error rate"))
    print("----------------------------------------------------------------------------------------------------------------")
    for K in Ks:
        for C in Cs:
            svm = SVM_Classifier()
            svm.train(DTR, LTR, C, K, factr=1.0, verbose=0)
            duality_gap, primal_loss, dual_loss = svm.duality_gap(DTR, LTR)
            duality_gap = float(duality_gap)
            primal_loss = float(primal_loss)
            dual_loss = float(dual_loss)
            pred_labels = svm.inference(DTE)
            correct_predictions = (pred_labels == LTE).sum()
            accuracy = correct_predictions / pred_labels.shape[0]
            print("{:>10}{:>10.1f}{:>20.6e}{:>20.6e}{:>20.6e}{:>20.1f}%".format(K, C, primal_loss, dual_loss, duality_gap, (1-accuracy)*100))
        print("----------------------------------------------------------------------------------------------------------------")



