import sys
import os
import numpy as np
import scipy.special

SCRIPT_PATH = os.path.dirname(__file__)
DS_INFERNO_PATH = os.path.join("..", "data", "inferno.txt")
DS_PURGATORIO_PATH = os.path.join("..", "data", "purgatorio.txt")
DS_PARADISO_PATH = os.path.join("..", "data", "paradiso.txt")

DS_INFERNO_PATH = os.path.join(SCRIPT_PATH, DS_INFERNO_PATH)
DS_PURGATORIO_PATH = os.path.join(SCRIPT_PATH, DS_PURGATORIO_PATH)
DS_PARADISO_PATH = os.path.join(SCRIPT_PATH, DS_PARADISO_PATH)

def load_data():
    lInf = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open(DS_INFERNO_PATH, encoding="ISO-8859-1")
    else:
        f = open(DS_INFERNO_PATH)

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open(DS_PURGATORIO_PATH, encoding="ISO-8859-1")
    else:
        f = open(DS_PURGATORIO_PATH)

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open(DS_PARADISO_PATH, encoding="ISO-8859-1")
    else:
        f = open(DS_PARADISO_PATH)
    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar


def split_data(l, n):
    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])

    return lTrain, lTest

def define_vocabulary(Dinf, Dpur, Dpar):
    def build_voc(voc, D):
        for line in D:
            words = line.split()
            for word in words:
                voc.add(word)
        return voc

    voc = set()
    build_voc(voc, Dinf)
    build_voc(voc, Dpur)
    build_voc(voc, Dpar)

    voc = list(voc)
    word_to_index = {word: i for i, word in enumerate(voc)}

    return voc, word_to_index


def preproc_dataset(D, word_to_index):
    # Build a vocabulary dictionary
    voc_occurences = {}
    for line in D:
        words = line.split()
        for word in words:
            if word in voc_occurences:
                voc_occurences[word] = voc_occurences[word] + 1
            else:
                voc_occurences[word] = 1

    # Prepare the dataset D (#vocabulary, #samples)
    Dpreproc = np.zeros((len(voc), D.shape[0]), dtype=int)
    for i, line in enumerate(D):
        words = line.split()
        for word in words:
            if (word not in word_to_index):
                continue
            row = word_to_index[word]

            Dpreproc[row, i] = Dpreproc[row, i] + 1

    voc_occ = np.zeros(len(voc))
    for word in voc_occurences:
        if (word not in word_to_index):
            continue
        voc_occ[word_to_index[word]] = voc_occurences[word]

    return Dpreproc, voc_occ

def discrete_linear_classifier_train(Dinf, Dpur, Dpar, voc_inf_train, voc_pur_train, voc_par_train, eps):
    Dinf_tot = Dinf.sum(axis=1)
    #print("Dinf tot shape: ", Dinf_tot.shape)
    Dinf_tot = Dinf_tot + eps
    Nc_inf = voc_inf_train.sum()

    #print(Dinf_tot[0:30])
    #print(voc_inf_train[0:30])
    pi_inf = Dinf_tot / Nc_inf

    Dpur_tot = Dpur.sum(axis=1)
    Dpur_tot = Dpur_tot + eps
    Nc_pur = voc_pur_train.sum()
    pi_pur = Dpur_tot / Nc_pur

    Dpar_tot = Dpar.sum(axis=1)
    Dpar_tot = Dpar_tot + eps
    Nc_par = voc_par_train.sum()

    pi_par = Dpar_tot / Nc_par

    pi = []
    pi.append(pi_inf)
    pi.append(pi_pur)
    pi.append(pi_par)

    return np.array(pi)

def discrete_lienar_classifier_inference(D, pi, Pc):
    log_likelihoods = np.log(pi) @ D
    #print("Log likelihoods shape: ", log_likelihoods.shape)
    Pc = np.array(Pc).reshape(3, 1)
    S_joint = log_likelihoods * Pc
    #print("Joint S matrix shape: ", S_joint.shape)
    marginal = scipy.special.logsumexp(S_joint, axis=0)
    S_posterior = S_joint - marginal
    pred_labels = np.argmax(S_posterior, 0)

    return pred_labels

if __name__ == '__main__':
    # Load the tercets and split the lists in training and test lists

    lInf, lPur, lPar = load_data()

    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    voc, word_to_index = define_vocabulary(lInf_train, lPur_train, lPar_train)

    lInf_train = np.array(lInf_train)
    lInf_evaluation = np.array(lInf_evaluation)

    lPur_train = np.array(lPur_train)
    lPur_evaluation = np.array(lPur_evaluation)

    lPar_train = np.array(lPar_train)
    lPar_evaluation = np.array(lPar_evaluation)

    lInf_train, vocInf_train = preproc_dataset(lInf_train, word_to_index)
    lInf_evaluation, vocInf_evaluation = preproc_dataset(lInf_evaluation, word_to_index)

    lPur_train, vocPur_train = preproc_dataset(lPur_train, word_to_index)
    lPur_evaluation, vocPur_evaluation = preproc_dataset(lPur_evaluation, word_to_index)

    lPar_train, vocPar_train = preproc_dataset(lPar_train, word_to_index)
    lPar_evaluation, vocPar_evaluation = preproc_dataset(lPar_evaluation, word_to_index)

    print("Inferno DS Train shape: ", lInf_train.shape, "; Test shape: ", lInf_evaluation.shape)
    print("Purgatorio DS Train shape: ", lPur_train.shape, "; Test shape: ", lPur_evaluation.shape)
    print("Paradiso DS Train shape: ", lPar_train.shape, "; Test shape: ", lPar_evaluation.shape)
    print("Inferno DS Train vocabulary shape: ", vocInf_train.shape, "; Test shape: ", vocInf_evaluation.shape)
    print("Purgatorio DS Train vocabulary  shape: ", vocPur_train.shape, "; Test shape: ", vocPur_evaluation.shape)
    print("Paradiso DS Train  vocabulary shape: ", vocPar_train.shape, "; Test shape: ", vocPar_evaluation.shape)

    #print("Vocabulary: ", voc[0:10])
    #print("Word to index: ", ["%s: %d" % (key, word_to_index[key]) for key in word_to_index.keys()][0:10])

    pi = discrete_linear_classifier_train(lInf_train, lPur_train, lPar_train, vocInf_train, vocPur_train, vocPar_train, 0.001)
    #print("Pi shape: ", pi.shape)

    Pc = [1/3, 1/3, 1/3]
    inf_preds = discrete_lienar_classifier_inference(lInf_evaluation, pi, Pc)
    inf_correct = (inf_preds == 0).sum()
    inf_acc = inf_correct / inf_preds.shape[0]
    print("Inferno accuracy: ", inf_acc)

    pur_preds = discrete_lienar_classifier_inference(lPur_evaluation, pi, Pc)
    pur_correct = (pur_preds == 1).sum()
    pur_acc = pur_correct / pur_preds.shape[0]
    print("Purgatorio accuracy: ", pur_acc)

    par_preds = discrete_lienar_classifier_inference(lPar_evaluation, pi, Pc)
    par_correct = (par_preds == 2).sum()
    par_acc = par_correct / par_preds.shape[0]
    print("Paradiso accuracy: ", par_acc)

    tot_correct = inf_correct + pur_correct + par_correct
    overall_acc = tot_correct / (inf_preds.shape[0] + pur_preds.shape[0] + par_preds.shape[0])
    print("Overall accuracy: ", overall_acc)
    # labels = np.hstack((np.zeros(lInf_evaluation.shape[1]), np.ones(lPur_evaluation.shape[1]), np.array([2 for i in range(lPar_evaluation.shape[1])])))
    # commedia_eval = np.concatenate((lInf_evaluation, lPur_evaluation, lPar_evaluation), axis=1)
    # commedia_preds = discrete_lienar_classifier_inference(commedia_eval, pi, Pc)
    # print("commedia preds shape: ", commedia_preds.shape)
    # print("labels shape: ", labels.shape)
    # correct = (commedia_preds == labels).sum()
    # print((commedia_preds == 1).sum())
    # acc = correct / commedia_preds.shape[0]
    # print("acc: ", acc)

