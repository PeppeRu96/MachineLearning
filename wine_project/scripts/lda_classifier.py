import numpy as np
import matplotlib.pyplot as plt

from preproc.dim_reduction.lda import lda
import preproc.dstools as dst
import wine_project.utility.ds_common as dsc
import evaluation.common as eval

# The goal is to project our 11-dimension feature space into a 1-dimension feature space starting from
# the gaussianized features, before using LDA, we try to discard none, one or two dimensions with the lowest variance
# through PCA, this to reduce the potential risk to have singular values on the Sw matrix.
# However, it's worth noting that starting from a 11-dimension feature space, maybe it would be better
# using only LDA without using PCA because discarding one or two dimensions regardless of the class may delete
# discriminant factors taken into account by LDA.
# LDA doesn't work...
if __name__ == "__main__":
    folds_data, folds_labels = dsc.load_train_dataset_5_folds()

    iterations = 1
    scores = []
    labels = []
    for DTR, LTR, DTE, LTE in dst.kfold_generate(folds_data, folds_labels):
        print("5-Fold Iteration ", iterations)
        DTR = dst.gaussianize_features(DTR, DTR)
        U, _ = lda(DTR, LTR, 1, False)

        DTE = dst.gaussianize_features(DTR, DTE)
        DTE_p = U.T @ DTE
        scores.append(DTE_p)
        labels.append(LTE)
        iterations += 1

    scores = np.array(scores)
    scores = scores.reshape((1, scores.shape[0]*scores.shape[2]))
    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0]*labels.shape[1]))

    # Threshold estimation
    # Shuffle the scores
    scores, labels = dst.shuffle_labeled_samples(scores, labels)

    # Splitting the scores in two partitions for estimating a good threshold
    (score_train, label_train), (score_eval, label_eval) = dst.split_db_2to1(scores, labels, 0.8)

    # Estimating threshold over the first partition
    estimated_minDCF, estimated_threshold = eval.bayes_min_dcf(score_train, label_train, dsc.applications[0][0], dsc.applications[0][1], dsc.applications[0][2], -10, 10, 1000)
    print("Estimated minDCF: ", estimated_minDCF)
    print("Estimated threshold: ", estimated_threshold)

    # Evaluating the goodness over the other partition
    evaluated_minDCF, _ = eval.bayes_min_dcf(score_eval, label_eval, dsc.applications[0][0], dsc.applications[0][1], dsc.applications[0][2], -10, 10, 1000)
    print("Min DCF on the validation partition: ", evaluated_minDCF)
    # Make the predictions using the estimated threshold
    predictions = eval.bayes_binary_optimal_classifier(score_eval, dsc.applications[0][0], dsc.applications[0][1], dsc.applications[0][2], estimated_threshold)
    conf_matr = eval.get_confusion_matrix(predictions, label_eval)
    actualDCF = eval.bayes_binary_dcf(conf_matr, dsc.applications[0][0], dsc.applications[0][1], dsc.applications[0][2])
    print("Actual DCF on the validation partition using the estimated threshold: ", actualDCF)

    # Draw ROC for the validation subset
    plt.figure()
    eval.draw_ROC(score_eval, label_eval, -10, 10, 1000)

    plt.figure()
    eval.draw_NormalizedBayesErrorPlot(score_eval, label_eval, -10, 10, 50, -10, 10, 1000, "LDA after Gaussianization")
    plt.show()
