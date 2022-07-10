import os
from os import walk
from shutil import copyfile
import numpy as np
from numpy import asarray
from PIL import Image


def main(source, dest_train, dest_test, training_ratio):
    # class names we want to study, you can add more
    classes = {}
    with open(os.path.join(source, 'labels.csv')) as f:
        for line in f:
            (k, v, q) = line.split(',')  # {'20160928-140314-0': ('0', 'Chiness apple')}
            classes[k] = (v, q)

    # generate output directories
    if not os.path.exists(dest_train):
        os.makedirs(dest_train)
    if not os.path.exists(dest_test):
        os.makedirs(dest_test)

    # get all the pictures in directory
    ext = (".JPEG", "jpeg", "JPG", ".jpg", ".png", "PNG")
    for (dirpath, dirnames, filenames) in walk(source):
        for filename in filenames:
            if filename.endswith(ext):
                # if less than ratio of training set, then put it to the training set
                if np.random.random(1)[0] <= training_ratio:
                    copyfile(os.path.join(source, filename),
                             os.path.join(dest_train, filename))
                else:
                    copyfile(os.path.join(source, filename),
                             os.path.join(dest_test, filename))

    # generate npz files for evaluating FID and IS
    images = []
    labels = []
    for (dirpath, dirnames, filenames) in walk(dest_test):
        for filename in filenames:
            image = Image.open(os.path.join(dest_test, filename))
            images.append(asarray(image))
            labels.append(int(classes[filename][0]))

    # np.save(dest_test + '/arr_0.npy', images)
    # np.save(dest_test + '/arr_1.npy', labels)
    np.savez('deepweeds.npz', np.array(images), np.array(labels))


if __name__ == "__main__":
    source = 'DeepWeeds'
    dest_train = 'DeepWeeds_train'
    dest_test = 'DeepWeeds_test'

    # ratio of training and testing
    training_ratio = 0.9

    main(source, dest_train, dest_test, training_ratio)
