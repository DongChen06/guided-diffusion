import os.path

import numpy as np
import matplotlib.pyplot as plt

class_index = ['Carpetweeds', 'Eclipta', 'Goosegrass', 'Morningglory',
               'Nutsedge', 'PalmerAmaranth', 'Purslane', 'Sicklepod', 'SpottedSpurge', 'Waterhemp']


def smooth(x, timestamps=9):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


# observation analysis
imgs_label = np.load('../datasets/cottonweed.npz')
dest_dir = '../model1/samples/'
os.makedirs(dest_dir, exist_ok=True)

# for debug usage
imags = imgs_label['arr_0']
labels = imgs_label['arr_1']

for index, img in enumerate(imgs_label['arr_0']):
    if not os.path.exists(dest_dir + class_index[imgs_label['arr_1'][index]]):
        os.mkdir(dest_dir + class_index[imgs_label['arr_1'][index]])
    plt.imsave(dest_dir + class_index[imgs_label['arr_1'][index]] + '/' + str(index) + '.jpg', img)
