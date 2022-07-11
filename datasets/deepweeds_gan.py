import os
from os import walk
from shutil import copyfile


def main(source, dest_train):
    # generate output directories
    os.makedirs(dest_train, exist_ok=True)

    # get all the pictures in directory
    for (dirpath, dirnames, filenames) in walk(source):
        for filename in filenames:
            os.makedirs(dest_train + '/' + filename.split('_')[0], exist_ok=True)
            copyfile(os.path.join(source, filename), os.path.join(dest_train, filename.split('_')[0], filename))


if __name__ == "__main__":
    source = 'DeepWeedsDiff_train'
    dest_train = 'DeepWeeds_train'

    # source_test = 'DeepWeedsDiff_test'
    # dest_test = 'DeepWeeds_test'

    main(source, dest_train)
    # main(source_test, dest_test)
