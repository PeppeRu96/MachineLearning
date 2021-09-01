import os
import numpy as np
import preproc.dstools as dst

SCRIPT_PATH = os.path.dirname(__file__)
DATASET_TRAIN_PATH = os.path.join(SCRIPT_PATH, "..", "data", "Train.txt")
DATASET_TEST_PATH = os.path.join(SCRIPT_PATH, "..", "data", "Test.txt")

FIVE_FOLDS_TRAIN_PATH = os.path.join(SCRIPT_PATH, "..", "data", "5-folds_train.npy")
FIVE_FOLDS_TRAIN_LABELS_PATH = os.path.join(SCRIPT_PATH, "..", "data", "5-folds_train_labels.npy")

# Here we discuss different applications where Priors = effective priors
# We don't care about priors and costs distinctly
applications = [
    (0.5, 1, 1),
    (0.1, 1, 1),
    (0.9, 1, 1)
]

def load_train_dataset():
    label_names = ["Low-quality (Hf)", "High-quality (Ht)"]
    feature_names = ["fixed acidity",
                     "volatile acidity",
                     "citric acid",
                     "residual sugar",
                     "chlorides",
                     "free sulfur dioxide",
                     "total sulfur dioxide",
                     "density",
                     "pH",
                     "sulphates",
                     "alcohol"
                     ]

    f = open(DATASET_TRAIN_PATH, "r")
    samples = []
    labels = []
    for line in f:
        line = line.strip().split(",")

        sample = [float(a) for a in line[:-1]]
        label = int(line[-1])

        samples.append(sample)
        labels.append(label)

    f.close()
    samples = np.array(samples).T
    labels = np.array(labels)
    dataset = dst.Dataset("Train Wine Dataset", label_names, feature_names, samples, labels)

    return dataset

def split_and_save_train_dataset_5_folds():
    # Load the train dataset in a wrapper of type Dataset to reuse useful utilities
    ds_train_wrapper = load_train_dataset()

    # We split our dataset in K folds only one time, so the different classifiers will be trained and validated
    # on the same set of folds, furthermore, saving the folds to the disk will speed up the process and
    # will leverage the possibility to train one model at a time, ensuring to not losing the initial split.
    # The folds will have the same class ratio as of the original dataset (0.33 for the Ht) to ensure that all the folds
    # will have samples of both classes;

    folds_data, folds_labels = dst.kfold_split(ds_train_wrapper.samples, ds_train_wrapper.labels, 5)

    np.save(FIVE_FOLDS_TRAIN_PATH, folds_data)
    np.save(FIVE_FOLDS_TRAIN_LABELS_PATH, folds_labels)

    return folds_data, folds_labels

def load_train_dataset_5_folds():
    folds_data = np.load(FIVE_FOLDS_TRAIN_PATH)
    folds_labels = np.load(FIVE_FOLDS_TRAIN_LABELS_PATH)

    return folds_data, folds_labels