import os
import sys
import numpy as np
import preproc.dstools as dst
from preproc.dim_reduction.lda import lda
from preproc.dim_reduction.pca import pca

import enum

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

def load_dataset(train, only_data=False):
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
    ds_path = DATASET_TRAIN_PATH if train else DATASET_TEST_PATH
    f = open(ds_path, "r")
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
    train_str = "Train" if train else "Test"
    dataset = dst.Dataset(f"{train_str} Wine Dataset", label_names, feature_names, samples, labels)
    if only_data:
        return dataset.samples, dataset.labels
    else:
        return dataset

def concat_kfolds(folds_data, folds_labels):
    folds_data_stack = np.hstack(folds_data)
    folds_labels_stack = np.hstack(folds_labels)

    return folds_data_stack, folds_labels_stack

def split_and_save_train_dataset_5_folds():
    # Load the train dataset in a wrapper of type Dataset to reuse useful utilities
    ds_train_wrapper = load_dataset(True)

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

class PreprocessConf:
    class PreprocStage:
        class Preproc(enum.Enum):
            Gaussianization = 1
            Centering = 2
            Z_Normalization = 3
            L2_Normalization = 4
            Whitening_Covariance = 5
            Whitening_Within_Covariance = 6

            @staticmethod
            def gaussianization(DTR, LTR, DTE):
                DTRoriginal = np.array(DTR)
                DTR = dst.gaussianize_features(DTRoriginal, DTR)
                DTE = dst.gaussianize_features(DTRoriginal, DTE)
                return DTR, DTE

            @staticmethod
            def centering(DTR, LTR, DTE):
                mu = DTR.mean(axis=1)
                mu = mu.reshape(mu.shape[0], 1)
                return DTR-mu, DTE-mu

            @staticmethod
            def z_normalization(DTR, LTR, DTE):
                DTRoriginal = np.array(DTR)
                DTR = dst.z_normalize(DTRoriginal, DTR)
                DTE = dst.z_normalize(DTRoriginal, DTE)
                return DTR, DTE

            @staticmethod
            def l2_normalization(DTR, LTR, DTE):
                DTR = dst.L2_normalize(DTR)
                DTE = dst.L2_normalize(DTE)
                return DTR, DTE

            @staticmethod
            def whitening_covariance(DTR, LTR, DTE):
                DTRoriginal = np.array(DTR)
                DTR = dst.whiten_covariance_matrix(DTRoriginal, DTR)
                DTE = dst.whiten_covariance_matrix(DTRoriginal, DTE)
                return DTR, DTE

            @staticmethod
            def whitening_within_covariance(DTR, LTR, DTE):
                DTRoriginal = np.array(DTR)
                DTR = dst.whiten_within_covariance_matrix(DTRoriginal, LTR, DTR)
                DTE = dst.whiten_within_covariance_matrix(DTRoriginal, LTR, DTE)
                return DTR, DTE

        map_funcs = {
            Preproc.Gaussianization: Preproc.gaussianization,
            Preproc.Centering: Preproc.centering,
            Preproc.Z_Normalization: Preproc.z_normalization,
            Preproc.L2_Normalization: Preproc.l2_normalization,
            Preproc.Whitening_Covariance: Preproc.whitening_covariance,
            Preproc.Whitening_Within_Covariance: Preproc.whitening_within_covariance
        }

        compact_strings = {
            Preproc.Gaussianization: "gau",
            Preproc.Centering: "center",
            Preproc.Z_Normalization: "znorm",
            Preproc.L2_Normalization: "l2norm",
            Preproc.Whitening_Covariance: "whitec",
            Preproc.Whitening_Within_Covariance: "whitewc"
        }

        def __init__(self, preproc):
            self.type = preproc

        def __str__(self):
            return "%s" % self.type.name

        def to_compact_string(self):
            return "%s" % self.__class__.compact_strings[self.type]

        def apply(self, DTR, LTR, DTE):
            DTR, DTE = self.__class__.map_funcs[self.type](DTR, LTR, DTE)
            return DTR, DTE


    class DimReductionStage:
        class DimRed(enum.Enum):
            Pca = 7
            Lda = 8

            @staticmethod
            def apply_pca(DTR, LTR, DTE, m):
                mu = DTR.mean(1)
                mu = mu.reshape(mu.size, 1)
                P, DTR, _ = pca(DTR, m)
                # Centering validation data
                DTE = DTE - mu
                DTE = P.T @ DTE
                return DTR, DTE

            @staticmethod
            def apply_lda(DTR, LTR, DTE, m):
                U, DTR = lda(DTR, LTR, m, False)
                DTE = U.T @ DTE
                return DTR, DTE

        map_funcs = {
            DimRed.Pca: DimRed.apply_pca,
            DimRed.Lda: DimRed.apply_lda
        }

        def __init__(self, dim_red, m):
            self.type = dim_red
            self.m = m

        def __str__(self):
            return "%s: %d" % (self.type.name, self.m)

        def to_compact_string(self):
            return "%s-%d" % (self.type.name.lower(), self.m)

        def apply(self, DTR, LTR, DTE):
            DTR, DTE = self.__class__.map_funcs[self.type](DTR, LTR, DTE, self.m)
            return DTR, DTE

    def __init__(self, preproc_stages):
        self.preproc_stages = preproc_stages

    def __str__(self):
        string = "[ %s" % self.preproc_stages[0] if len(self.preproc_stages) == 1 else "[ "
        for s in self.preproc_stages[:-1]:
            string = string + "%s, " % s
        string = string + "%s ]" % (self.preproc_stages[-1] if len(self.preproc_stages) > 1 else "")
        return string

    def to_compact_string(self):
        string = "%s" % self.preproc_stages[0].to_compact_string() if len(self.preproc_stages) == 1 else ""
        for s in self.preproc_stages[:-1]:
            string = string + "%s_" % s.to_compact_string()
        string = string + "%s" % (self.preproc_stages[-1].to_compact_string() if len(self.preproc_stages) > 1 else "")
        return string

    def apply_preproc_pipeline(self, DTR, LTR, DTE):
        for stage in self.preproc_stages:
            DTR, DTE = stage.apply(DTR, LTR, DTE)
        return DTR, DTE

# Shorthands
PreprocStage = PreprocessConf.PreprocStage
Preproc = PreprocStage.Preproc
DimReductionStage = PreprocessConf.DimReductionStage
DimRed = DimReductionStage.DimRed

class LoggingPrinter:
    def __init__(self, filename):
        self.out_file = open(filename, "w", encoding="utf-8")
        self.old_stdout = sys.stdout
        sys.stdout = self
    def write(self, text):
        self.old_stdout.write(text)
        self.out_file.write(text)
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_stdout

    original_stdout = sys.stdout  # Save a reference to the original standard output

def incremental_path(basepath, filename):
    complete_path = os.path.join(basepath, filename)
    if not os.path.exists(complete_path):
        return complete_path

    target_f = filename.split(".")
    target_fname = target_f[-2]
    target_fext = target_f[-1]
    target_f = target_fname.split("_")
    target_fbasename = target_f[:-1]
    target_fbasename = "_".join(target_fbasename)
    target_fid = int(target_f[-1])

    onlyfiles = [f for f in os.listdir(basepath) if os.path.isfile(os.path.join(basepath, f))]
    for f in onlyfiles:
        f = f.split(".")
        fname = f[-2]
        fext = f[-1]
        f = fname.split("_")
        fbasename = f[:-1]
        fbasename = "_".join(fbasename)
        if fbasename == target_fbasename:
            fid = int(f[-1])
            if fid >= target_fid:
                target_fid = fid + 1

    newfname = "%s_%d.%s" % (target_fbasename, target_fid, target_fext)

    return os.path.join(basepath, newfname)

def create_folder_if_not_exist(file_path):
    head, tail = os.path.split(file_path)
    if not os.path.isdir(head):
        os.mkdir(head)
