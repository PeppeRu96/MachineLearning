import numpy as np
import matplotlib.pyplot as plt
import enum
from typing import List

DEBUG = 0

class VISUALIZE(enum.Enum):
    All = 1
    Specified = 2
    Hidden = 3


# DATASET WRAPPER WITH TOOLS AND UTILITIES
# SAMPLES ARE STORED AS COLUMN VECTORS INSIDE A GLOBAL NDARRAY D WITH SHAPE (#FEATURES, #SAMPLES)
# LABELS ARE STORED AS A SINGLE ROW VECTOR (NDARRAY WITH SHAPE (#SAMPLES,) PAY ATTENTION AT THE SHAPE!)
class Dataset:

    def __init__(self, name: str, label_names: List[str], feature_names: List[str], samples: np.ndarray,
                 labels: np.ndarray) -> None:
        self.name = name
        self.label_names = label_names
        self.label_to_val = {l: idx for idx, l in enumerate(label_names)}
        self.feature_names = feature_names
        self.samples = samples
        self.labels = labels
        if DEBUG:
            print("Dataset %s" % name)
            print("samples shape: ", samples.shape)
            print("labels shape: ", labels.shape)

            print("Number of labels: %d" % (len(label_names)))
            print("Labels: ", label_names)
            print("Val_To_labels: ", self.label_to_val)
            print("Number of features: %d" % (samples.shape[0]))
            print("Features: ", feature_names)
            print("Number of samples: %d" % (samples.shape[1]))

    def visualize_histogram(self, show: VISUALIZE = VISUALIZE.All, features_to_show: List[str] = None) -> None:
        visualize_histogram(self.samples, self.labels, self.feature_names, self.label_names, show, features_to_show)

    def visualize_scatter(self, show: VISUALIZE = VISUALIZE.All, features_to_show: List[str] = None) -> None:
        visualize_scatter(self.samples, self.labels, self.feature_names, self.label_names, show, features_to_show)

    def split_db_2to1(self, train_fraction, seed=0):
        split_db_2to1(self.samples, self.labels, train_fraction, seed)


def load_iris_from_csv(file_path: str) -> Dataset:
    label_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    labels_to_val = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    feature_names = ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"]

    f = open(file_path, "r")
    samples = []
    labels = []
    for line in f:
        line = line.strip().split(",")
        if len(line) != 5:
            continue

        sample = [float(a) for a in line[:-1]]
        label = labels_to_val[line[-1]]

        samples.append(sample)
        labels.append(label)

    f.close()
    samples = np.array(samples).T
    labels = np.array(labels)
    dataset = Dataset("Iris", label_names, feature_names, samples, labels)

    return dataset


# ds: Dataset wrapper object
# hist_show: should pass an HISTOGRAM_SHOW type
# features_to_show: should pass a list of valid feature indices
def visualize_histogram(D: np.ndarray, L: np.ndarray, feature_names: List[str], label_names: List[str],
                        show: VISUALIZE = VISUALIZE.All, features_to_show: List[str] = None, bins = None) -> None:
    if features_to_show is None:
        features_to_show = []
    if show == VISUALIZE.Hidden:
        return None
    if show == VISUALIZE.All:
        features_to_show = feature_names
    else:
        feats = []
        for i, f in enumerate(feature_names):
            if i in features_to_show:
                feats.append(feature_names[i])
            else:
                feats.append(None)
        features_to_show = feats

    # For each feature
    for idx_feat, name_feat in enumerate(features_to_show):
        if name_feat is None:
            continue

        if DEBUG:
            print("Showing histogram for the feature '%s'.." % name_feat)
        plt.figure()
        plt.title(name_feat)
        plt.xlabel(name_feat)
        # For each label
        for i, label_name in enumerate(label_names):
            Mi = (L == i)  # Mask
            # Filter columns (selects only the cols, that are the samples which belong to the current label)
            # and select the row of the curr attribute.
            Di = D[idx_feat, Mi]
            plt.hist(Di, density=True, label=label_name, alpha=0.5, bins=bins)
        plt.legend()


# Visualize, for the given features, all the pairs in scatter plots
def visualize_scatter(D: np.ndarray, L: np.ndarray, feature_names: List[str], label_names: List[str],
                      show: VISUALIZE = VISUALIZE.All, features_to_show: List[str] = None) -> None:
    if show == VISUALIZE.Hidden:
        return None
    if show == VISUALIZE.All:
        features_to_show = feature_names
    else:
        feats = []
        for i, f in enumerate(feature_names):
            if i in features_to_show:
                feats.append(feature_names[i])
            else:
                feats.append(None)
        features_to_show = feats

    for idx_feat_x, name_feat_x in enumerate(features_to_show):
        if name_feat_x is None:
            continue
        for idx_feat_y, name_feat_y in enumerate(features_to_show):
            if name_feat_y is None or idx_feat_y <= idx_feat_x:
                continue

            if DEBUG:
                print("Showing scatter plot for the pair (%s(%d), %s(%d)).." % (
                    name_feat_x, idx_feat_x, name_feat_y, idx_feat_y))
            plt.figure()
            plt.title("Scatter plot comparing %s to %s" % (name_feat_x, name_feat_y))
            plt.xlabel(name_feat_x)
            plt.ylabel(name_feat_y)
            for i, label_name in enumerate(label_names):
                Mi = (L == i)
                Dix = D[idx_feat_x, Mi]
                Diy = D[idx_feat_y, Mi]
                plt.scatter(Dix, Diy, label=label_name)
            plt.legend()

# Splits a dataset into a train dataset and test dataset according to the train_fraction with random extracted samples
def split_db_2to1(D, L, train_fraction, seed=0):
    print("Splitting dataset in %f train data and %f test data" % (train_fraction, (1-train_fraction)))
    nTrain = int(D.shape[1] * train_fraction)
    print("Train samples: %d; Test samples: %d" % (nTrain, D.shape[1]-nTrain))
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# Split a dataset (D, L) in K folds equally distributed for each class
# It's based on the assumption that the dataset has a number of samples equal for each class
# Otherwise, it should be required some addition logic to build more robust folds
# Returns two numpy arrays:
# folds_data (K, #dimensions, #SAMPLES/K)
# folds_labels (K, #SAMPLES/K)
def kfold_split(D, L, K):
    if DEBUG:
        print("K FOLDS SPLIT WITH K=%d" % K)
    Dcopy = np.array(D)
    samples_cnt = D.shape[1]
    fold_size = samples_cnt // K
    classes = len(set(L))
    #fold_size_class = fold_size / classes

    Dclass = []
    indexes = []
    for i in range(classes):
        Di = Dcopy[:, (L==i)]
        Dclass.append(Di)

        idx = list(np.random.permutation(Di.shape[1]))
        indexes.append(idx)

    Dclass = np.array(Dclass)

    if DEBUG:
        print("Dataset suvdivided for each class: ", Dclass.shape)

    folds_data = []
    folds_labels = []
    for i in range(K):
        c = 0
        fold_data = []
        fold_labels = []
        for j in range(fold_size):
            while len(indexes[c]) <= 0:
                c = (c + 1) % classes
            id = indexes[c].pop(0)
            sample = Dclass[c, :, id]
            fold_data.append(sample)
            fold_labels.append(c)
            c = (c + 1) % classes

        fold_data = np.array(fold_data).T
        fold_labels = np.array(fold_labels)

        # Permutate randomly samples inside the fold
        fold_idx = np.random.permutation(fold_data.shape[1])
        fold_data = fold_data[:, fold_idx]
        fold_labels = fold_labels[fold_idx]

        folds_data.append(fold_data)
        folds_labels.append(fold_labels)

    folds_data = np.array(folds_data)
    folds_labels = np.array(folds_labels)
    if DEBUG:
        print("Folds data np array shape: ", folds_data.shape)
        print("Folds labels np array shape: ", folds_labels.shape)

    return folds_data, folds_labels

# Requires folds (K, #dimensions, #samples), folds_labels (#folds, #samples)
# Returns a generator that can be looped with DTR, LTR, DTE, LTE up to K times
def kfold_generate(folds, folds_labels):
    for i in range(folds.shape[0]):
        folds_copy = list(folds)
        folds_labels_copy = list(folds_labels)

        DTE = np.array(folds_copy.pop(i))
        LTE = np.array(folds_labels_copy.pop(i))

        folds_copy = np.array(folds_copy)
        folds_labels_copy = np.array(folds_labels_copy)
        DTR = folds_copy[0]
        for j in range(1, folds_copy.shape[0]):
            DTR = np.concatenate((DTR, folds_copy[j]), axis=1)

        LTR = folds_labels_copy.flatten()

        yield DTR, LTR, DTE, LTE