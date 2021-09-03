import os
import numpy as np
import matplotlib.pyplot as plt

from preproc.dim_reduction.pca import pca
import preproc.dstools as dst
from wine_project.utility.ds_common import *
import evaluation.common as eval

import time

if __name__ == "__main__":
    # Load 5-Folds already splitted dataset
    folds_data, folds_labels = load_train_dataset_5_folds()

    # Define preprocessing configurations (to be cross-validated as different models)
    preproc_conf = PreprocessConf([
        PreprocStage(Preproc.Gaussianization),
        PreprocStage(Preproc.Centering),
        PreprocStage(Preproc.Z_Normalization),
        PreprocStage(Preproc.L2_Normalization),
        PreprocStage(Preproc.Whitening_Covariance),
        PreprocStage(Preproc.Whitening_Within_Covariance),
        DimReductionStage(DimRed.Pca, 10),
        DimReductionStage(DimRed.Lda, 10)
    ])


    iterations = 1
    scores = []
    labels = []
    for DTR, LTR, DTE, LTE in dst.kfold_generate(folds_data, folds_labels):
        print("5-Fold Iteration ", iterations)
        print(preproc_conf.to_compact_string())
        #DTR, DTE = preproc_conf.apply_preproc_pipeline(DTR, LTR, DTE)

        iterations +=1


