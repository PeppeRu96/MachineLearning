import wine_project.utility.ds_common as dsc
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch dataset preprocessing",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--load", default=True, action="store_true", help="Load already splitted dataset")

    return parser.parse_args()

if __name__ == "__main__":
    # Pay attention: don't run the split, the goal is to have the same 5 folds
    # for all the training stage (all classifiers)
    # We split our dataset in 5 folds only one time, so the different classifiers will be trained and validated
    # on the same set of folds, furthermore, saving the folds to the disk will speed up the process and
    # will leverage the possibility to train one model at a time, ensuring to not losing the initial split.
    # The folds will have the same class ratio as of the original dataset (0.33 for the Ht) to ensure that all the folds
    # will have samples of both classes;

    args = get_args()

    if args.load is False:
        folds_data, folds_labels = dsc.split_and_save_train_dataset_5_folds()
    else:
        folds_data, folds_labels = dsc.load_train_dataset_5_folds()
    print("Folds data shape: ", folds_data.shape)
    print("Folds labels shape: ", folds_labels.shape)
    print("")
