import os.path
import numpy as np
import matplotlib.pyplot as plt


# class names we want to study, you can add more
class_index = ['Chineeapple', 'Lantana', 'Parkinsonia', 'Parthenium',
               'Pricklyacacia', 'Rubbervine', 'Siamweed', 'Snakeweed', 'Negative']

# observation analysis
imgs_label = np.load('../model256_deepweeds/samples_2000x256x256x3_ADM.npz')
dest_dir = '../model256_deepweeds/samples_class_ADM/'
dest_dir_separate = '../model256_deepweeds/samples_class_separate_ADM/'
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(dest_dir_separate, exist_ok=True)

# for debug usage
# imags = imgs_label['arr_0']
# labels = imgs_label['arr_1']

for index, img in enumerate(imgs_label['arr_0']):
    if not os.path.exists(dest_dir_separate + class_index[imgs_label['arr_1'][index]]):
        os.mkdir(dest_dir_separate + class_index[imgs_label['arr_1'][index]])
        # save images according to their classes
    plt.imsave(dest_dir_separate + class_index[imgs_label['arr_1'][index]] + '/' + str(index) + '.jpg', img)

    # save images in one folder
    plt.imsave(dest_dir + '/' + str(index) + '.jpg', img)
