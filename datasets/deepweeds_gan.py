import os
from os import walk
from shutil import copyfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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


def main(source, dest_train_fid, dest_train_gan, image_size):
    # generate output directories
    os.makedirs(dest_train_fid, exist_ok=True)
    os.makedirs(dest_train_gan, exist_ok=True)

    # save images for calculating FID, improved precision and recall
    for (dirpath, dirnames, filenames) in walk(source):
        for filename in filenames:
            image = Image.open(os.path.join(source, filename))
            image = center_crop_arr(image, image_size)
            plt.imsave(dest_train_fid + '/' + filename, image)

    # save images to train GANs
    for (dirpath, dirnames, filenames) in walk(source):
        for filename in filenames:
            os.makedirs(dest_train_gan + '/' + filename.split('_')[0], exist_ok=True)
            copyfile(os.path.join(source, filename), os.path.join(dest_train_gan, filename.split('_')[0], filename))


if __name__ == "__main__":
    source = 'DeepWeedsDiff_test'
    dest_train_fid = 'DeepWeeds_test_fid'
    dest_train_gan = 'DeepWeeds_test_gan'
    image_size = 256
    # source_test = 'DeepWeedsDiff_test'
    # dest_test = 'DeepWeeds_test'

    main(source, dest_train_fid, dest_train_gan, image_size)
    # main(source_test, dest_test)
