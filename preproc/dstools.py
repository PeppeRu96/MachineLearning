import numpy as np
import matplotlib.pyplot as plt
import enum

DEBUG = 0


class VISUALIZE(enum.Enum):
    All = 1
    Specified = 2
    Hidden = 3


# DATASET WRAPPER WITH TOOLS AND UTILITIES
# SAMPLES ARE STORED AS COLUMN VECTORS INSIDE A GLOBAL NDARRAY D WITH SHAPE (#FEATURES, #SAMPLES)
# LABELS ARE STORED AS A SINGLE ROW VECTOR (NDARRAY WITH SHAPE (#SAMPLES,) PAY ATTENTION AT THE SHAPE!)
class Dataset:

    def __init__(self, name, label_names, feature_names, samples, labels):
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

    def visualize_histogram(self, show=VISUALIZE.All, features_to_show = []):
        visualize_histogram(self.samples, self.labels, self.feature_names, self.label_names, show, features_to_show)

    def visualize_scatter(self, show=VISUALIZE.All, features_to_show = []):
        visualize_scatter(self.samples, self.labels, self.feature_names, self.label_names, show, features_to_show)


def load_iris_from_csv(file_path):
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
def visualize_histogram(D, L, feature_names, label_names, show=VISUALIZE.All, features_to_show=[]):
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
            plt.hist(Di, density=True, label=label_name, alpha=0.5)
        plt.legend()

# Visualize, for the given features, all the pairs in scatter plots
def visualize_scatter(D, L, feature_names, label_names, show=VISUALIZE.All, features_to_show=[]):
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
                print("Showing scatter plot for the pair (%s(%d), %s(%d)).." % (name_feat_x, idx_feat_x, name_feat_y, idx_feat_y))
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
