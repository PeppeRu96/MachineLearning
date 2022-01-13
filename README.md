# Wine Quality Detection ML Project
Author: Giuseppe Ruggeri, s292459
## Folder structure
The folder structure is very simple and it is composed as following:
- classifiers - a package containing all the needed classifiers
- density_estimation - a package containing the density estimation methods
- evaluation - a package containing the methods for evaluating our models
- preproc - a package containing all the preprocess methods
- wine_project - a folder containing all the scripts, logs, and graphs for the project
  - utility - a package containing recurrent methods to be used during the analyses
  - data - contains the dataset and the cached models for fast processing
  - graphs - will contain the generated graphs
  - train_logs - will contain the generated train logs
  - scripts - it contains all the scripts for performing the different analyses
 
## Usage
Please notice that you have to add the modules to your PYTHONPATH variable or use a sufficiently intelligent IDE (e.g. PyCharm).
We briefly list, for each available analysis, the available options.  
**NB: For the classifiers are available all the option repeated in three variants: standard, with the prefix *eval_partial_* and with the prefix *eval_full_*, they are omitted in the following for conciseness.  
The eval_partial_ starts the same functionality but on the evaluation dataset, using 4/5 of the training dataset.  
The eval_full_ starts the same functionality but on the evaluation dataset, using the full training dataset.**
### ds_preprocessing.py
It just splits the dataset generating the 5-folds split
- --load (default: true) - load the already 5-folds split dataset

## ds_analysis.py
Performs the training dataset analysis, including raw and gaussian features, histograms, correlation matrices.
- --hist_raw (default: false) - show histograms for the raw featuers
- --hist-gau (default: false) - show histograms for the gaussianized features
- --show_correlations (defualt: false) - show correlation matrices

## mvg_classifier.py
Performs the model selection, hyperparameters optimization on the train dataset and evaluation contrasting on the test dataset.
- --gridsearch (default: false) - perform a gridsearch on the train dataset
- --eval_partial_..
- --eval_full_..

## lr_classifier.py
- --linear (default: false) - performs a gridsearch for linear LR on the train dataset
- --quadratic (default: false) - performs a gridsearch for quadratic LR on the train dataset
- --linear_app_specific (default: false) - embed the target priors in the objective for the best models obtained during the gridsearch
- --quadratic_app_specific (default: false) - embed the target priors in the objective for the best models obtained during the gridsearch
- --eval_partial_..
- --eval_full_..

## linear_svm_classifier.py
- --gridsearch (default: false) - performs a gridsearch for linear SVM on the train dataset
- --class_balancing (default: false) - performs a small gridsearch (with different values of C and the best preproc configuration) but class-balancing with respect to the target application
- --eval_partial_..
- --eval_full_..

## polynomial_svm_classifier.py
- --gridsearch (default: false) - performs a gridsearch for quadratic and cubic SVM on the train dataset
- --class_balancing (default: false) - performs a class-balancing training with the best preproc configuration and the best hyperparameters found in the gridsearch
- --eval_partial_..
- --eval_full_..

## rbf_svm_classifier.py
- --gridsearch (default: false) - performs a coarse-level gridsearch for rbf svm to jointly optimize C and gamma
- --gridsearch_fine_grained (default: false) - performs a fine-grained gridsearch around the optimal values found
- --class_balancing (default: false) - performs a class-balancing training with the best preproc configuration and the best hyperparameters found in the gridseach
- --actual_dcf (default: false) - compute the actual dcf on the validation set (of the pulled scores during the K-fold cross-validation on the train set)
- --eval_partial_..
- --eval_full_..

## gmm_classifier.py
- --gridsearch (default: false) - start a gridsearch cross-validation to optimize with respect to the number of components, the preprocess configuration and the model type
- --actual_dcf (default: false) - compute the actual dcf on the validation set (of the pulled scores during the K-fold cross-validation on the train set)
- --eval_partial_..
- --eval_full_..

## svm_gmm_analysis.py
Performs the analysis of the two candidate models, re-calibration and fusion.
- --bayes_error_plot (default: false) - shows the bayes error plots of the two candidate models (uncalibrated) over the validation dataset (of the train dataset)
- --lr_recalibration (default: false) - recalibrate the scores using a linear LR for the best svm and gmm model and fuse them
- --eval_partial_..
- *NB: eval_full_ not available*
