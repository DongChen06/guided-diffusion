import os
from os import walk
from shutil import copyfile
import numpy as np
from numpy import asarray
from PIL import Image


def main(source, dest_train):
    # generate output directories
    os.makedirs(dest_train, exist_ok=True)

    # get all the pictures in directory
    for (dirpath, dirnames, filenames) in walk(source):
        for filename in filenames:
            os.makedirs(dest_train + '/' + filename.split('_')[0], exist_ok=True)
            copyfile(os.path.join(source, filename), os.path.join(dest_train, filename.split('_')[0], filename))


if __name__ == "__main__":
    source = 'CottonWeedDiff_train'
    dest_train = 'CottonWeed_train'

    source_test = 'CottonWeedDiff_test'
    dest_test = 'CottonWeed_test'

    main(source, dest_train)
    main(source_test, dest_test)
