import os
import matplotlib.pyplot as plt

import preproc.dstools as dst
import wine_project.utility.ds_common as dsc

if __name__ == "__main__":
    # Load the train dataset in a wrapper of type Dataset to reuse useful utilities
    ds_train_wrapper = dsc.load_train_dataset()

    ds_train_wrapper.visualize_statistics()


