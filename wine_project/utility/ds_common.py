import os
import numpy as np
import preproc.dstools as dst

SCRIPT_PATH = os.path.dirname(__file__)
DATASET_TRAIN_PATH = os.path.join(SCRIPT_PATH, "..", "data", "Train.txt")
DATASET_TEST_PATH = os.path.join(SCRIPT_PATH, "..", "data", "Test.txt")

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
