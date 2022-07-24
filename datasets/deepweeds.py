import os
from os import walk
from shutil import copyfile
import numpy as np
from numpy import asarray
from PIL import Image


# class names we want to study, you can add more
class_index = ['Chineeapple', 'Lantana', 'Parkinsonia', 'Parthenium',
               'Pricklyacacia', 'Rubbervine', 'Siamweed', 'Snakeweed', 'Negative']


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main(source, dest_train, dest_test, training_ratio, image_size):
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

    index_train, index_test = 0, 0
    # get all the pictures in directory
    ext = (".JPEG", "jpeg", "JPG", ".jpg", ".png", "PNG")
    for (dirpath, dirnames, filenames) in walk(source):
        for filename in filenames:
            if filename.endswith(ext):
                # get the image class and remove the space in the name
                cls = classes[filename][1].replace(" ", "").strip()
                # if less than ratio of training set, then put it to the training set
                if np.random.random(1)[0] <= training_ratio:
                    image_name = cls + '_' + str(index_train) + '.jpg'
                    copyfile(os.path.join(source, filename),
                             os.path.join(dest_train, image_name))
                    index_train += 1
                else:
                    image_name = cls + '_' + str(index_train) + '.jpg'
                    copyfile(os.path.join(source, filename),
                             os.path.join(dest_test, image_name))
                    index_test += 1

    # generate npz files for evaluating FID and IS
    images = []
    labels = []
    for (dirpath, dirnames, filenames) in walk(dest_train):
        for filename in filenames:
            image = Image.open(os.path.join(dest_train, filename))
            image = center_crop_arr(image, image_size)
            images.append(asarray(image))
            labels.append(class_index.index(filename.split('_')[0]))

    # np.save(dest_test + '/arr_0.npy', images)
    # np.save(dest_test + '/arr_1.npy', labels)
    np.savez('deepweeds.npz', np.array(images), np.array(labels))


if __name__ == "__main__":
    source = 'DeepWeeds'
    dest_train = 'DeepWeedsDiff_train'
    dest_test = 'DeepWeedsDiff_test'

    # ratio of training and testing
    training_ratio = 0.9
    image_size = 256

    main(source, dest_train, dest_test, training_ratio, image_size)
