import numpy as np
import matplotlib.pyplot as plt
import enum
from typing import List

from scipy.stats import norm

from preproc.dim_reduction.lda import sb_sw_compute

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

    def deep_copy(self):
        ds = Dataset(self.name, self.label_names, self.feature_names, np.array(self.samples), np.array(self.labels))
        return ds

    def visualize_statistics(self):
        print("Dataset: %s" % self.name)
        print("Number of different features: %d" % len(self.feature_names))
        print("Feature names: ", self.feature_names)
        print("Labels:")
        for i, l in enumerate(self.label_names):
            print("\t(%d) %s" % (i, l))
        print("")
        print("Number of samples: %d" % self.samples.shape[1])
        print("Number of samples for each label:")
        for i, l in enumerate(self.label_names):
            L = self.labels
            Di = self.samples[:, (L==i)]
            print("%s: %d" %(l, Di.shape[1]))

        if len(self.label_names) == 2:
            D = self.samples
            L = self.labels
            Ht_empirical_ratio = (D[:, (L==1)].shape[1]) / (D.shape[1])
            print("Ht (True class) empirical dataset prior probability: %.2f" % Ht_empirical_ratio)
        print("")
        # Statistics on features
        print("Statistics on features:")
        mins = self.samples.min(axis=1)
        maxs = self.samples.max(axis=1)
        means = self.samples.mean(axis=1)
        means_col = means.reshape((means.shape[0], 1))
        samples_centered = self.samples - means_col
        variances = (samples_centered * samples_centered).sum(axis=1) / self.samples.shape[1]
        print("{:>20}{:>20}{:>20}{:>20}{:>20}".format("Feature", "Min", "Max", "Mean", "Var"))
        print("------------------------------------------------------------------------------------------------------------")
        for f in range(len(self.feature_names)):
            print("{:>20}{:>20.4f}{:>20.4f}{:>20.4f}{:>20.8f}".format(self.feature_names[f], mins[f], maxs[f], means[f], variances[f]))


    def visualize_histogram(self, show: VISUALIZE = VISUALIZE.All, features_to_show: List[int] = None, bins=None,
                            separate_classes=False, base_title="", save_path=None) -> None:
        visualize_histogram(self.samples, self.labels, self.feature_names, self.label_names, show, features_to_show,
                            bins, separate_classes, base_title, save_path)

    def visualize_scatter(self, show: VISUALIZE = VISUALIZE.All, features_to_show: List[int] = None) -> None:
        visualize_scatter(self.samples, self.labels, self.feature_names, self.label_names, show, features_to_show)

    def split_db_2to1(self, train_fraction, seed=0):
        split_db_2to1(self.samples, self.labels, train_fraction, seed)

    def gaussianize_features(self, DRAW):
        Y = gaussianize_features(self.samples, DRAW)
        Yds = Dataset(self.name + " Gaussianized", self.label_names, self.feature_names, Y, self.labels)
        return Yds


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
                        show: VISUALIZE = VISUALIZE.All, features_to_show: List[int]=None, bins=None,
                        separate_classes=False, base_title="", save_path=None) -> None:
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
        plt.title(f"{base_title} - Feature: {name_feat}")
        plt.xlabel(name_feat)
        if separate_classes:
            # For each class
            for i, label_name in enumerate(label_names):
                Mi = (L == i)  # Mask
                # Filter columns (selects only the cols, that are the samples which belong to the current label)
                # and select the row of the curr attribute.
                Di = D[idx_feat, Mi]
                plt.hist(Di, density=True, label=label_name, alpha=0.5, bins=bins, histtype='bar', ec='black')
                plt.legend()
        else:
            Di = D[idx_feat, :]
            plt.hist(Di, density=True, alpha=0.5, bins=bins, histtype='bar', ec='black')

        if save_path is not None:
            sep_class_str = "sep-class" if separate_classes else "all"
            plt.savefig("%s_hist_%d_%s_%s" % (save_path, idx_feat, name_feat, sep_class_str))


# Visualize, for the given features, all the pairs in scatter plots
def visualize_scatter(D: np.ndarray, L: np.ndarray, feature_names: List[str], label_names: List[str],
                      show: VISUALIZE = VISUALIZE.All, features_to_show: List[int] = None) -> None:
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
    idx = np.random.RandomState(seed=seed).permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# Split a dataset (D, L) in K folds equally distributed for each class
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

    # Obtaining classses ratios to build folds with equal number of samples for each class as the original dataset
    classsesRatios = np.zeros(classes)
    for i in range(classes):
        ratio = D[:, (L==i)].shape[1] / D.shape[1]
        classsesRatios[i] = ratio


    Dclass = []
    indexes = []
    for i in range(classes):
        Di = Dcopy[:, (L==i)]
        Dclass.append(Di)

        idx = list(np.random.permutation(Di.shape[1]))
        indexes.append(idx)

    folds_data = []
    folds_labels = []
    for i in range(K):
        c = 0
        fold_data = []
        fold_labels = []
        count_per_class = np.zeros(classes)
        for j in range(fold_size):
            while len(indexes[c]) <= 0:
                c = (c + 1) % classes
            id = indexes[c].pop(0)
            sample = Dclass[c][:, id]
            fold_data.append(sample)
            fold_labels.append(c)
            count_per_class[c] += 1
            # Switch to pick a sample for the class whose fold ratio is the most different than the requested one
            ratios = count_per_class / len(fold_data)
            nextC = np.argmax(classsesRatios - ratios)
            c = nextC

        fold_data = np.array(fold_data).T
        fold_labels = np.array(fold_labels)

        # Permutate randomly samples inside the fold
        fold_data, fold_labels = shuffle_labeled_samples(fold_data, fold_labels)

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

def gaussianize_features(DTR, DRAW):
    Y = np.zeros((DRAW.shape[0], DRAW.shape[1]))
    for f in range(DTR.shape[0]):
        DTRf = DTR[f, :]
        DRAWf = DRAW[f, :]
        # For each raw point
        for i in range(DRAWf.shape[0]):
            x = DRAWf[i]
            rank = ((DTRf > x).sum() + 1) / (DTR.shape[1] + 2)
            y = norm.ppf(rank)
            Y[f, i] = y

    return Y

def z_normalize(DTR, DRAW):
    # Compute the means for all dimensions
    mu = DTR.mean(axis=1)
    mu = mu.reshape(mu.shape[0], 1)

    # Compute all standard deviations
    std = np.diag(covariance_matrix(DTR)) ** 0.5
    std = std.reshape(std.shape[0], 1)

    return (DRAW - mu) / std

def L2_normalize(D):
    norm = np.linalg.norm(D, ord=2, axis=0)

    return D / norm

def whiten_covariance_matrix(DTR, DRAW):
    cov = covariance_matrix(DTR)
    s, U = np.linalg.eigh(cov)
    A = U @ np.diag(1.0 / (s ** 0.5)) @ U.T
    #A = np.dot(np.dot(U, np.diag(1.0 / (s ** 0.5))), U.T)

    D_whiten = A @ DRAW
    return D_whiten

def whiten_within_covariance_matrix(DTR, LTR, DRAW):
    _, Sw = sb_sw_compute(DTR, LTR)

    U, s, _ = np.linalg.svd(Sw)
    A = np.dot(np.dot(U, np.diag(1.0 / (s ** 0.5))), U.T)

    D_whiten = A @ DRAW
    return D_whiten


def covariance_matrix(D):
    mu = D.mean(axis=1)
    mu = mu.reshape(mu.shape[0], 1)
    tmp = D - mu
    cov = (tmp @ tmp.T) / D.shape[1]

    return cov

def correlation_matrix(D):
    cov = covariance_matrix(D)
    corr = np.array(cov)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            std_x = cov[i, i] ** 0.5
            std_y = cov[j, j] ** 0.5
            corr[i, j] = corr[i, j] / (std_x * std_y)

    return corr

def shuffle_labeled_samples(D, L):
    idx = np.random.permutation(D.shape[1])
    Dshuffled = D[:, idx]
    Lshuffled = L[idx]

    return Dshuffled, Lshuffled