import getopt
import glob
import os
import sys
import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main(argv):
    """
    Program to create a model.

    :param argv:
        -h, --help:     Get help on input parameters.
        -f, --folder:   Define input folder that contains the training data. [C:/images]
        -e, --ext:      Define extension of image files. [.png]
    :return:
    """

    root_dir = 'C:/images'  # default folder
    ext = '.png'

    try:
        opts, args = getopt.getopt(argv, "hf:e:", ["help", "folder=", "ext="])
    except getopt.GetoptError:
        print('Invalid input parameter.')
        print(main.__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(main.__doc__)
        elif opt in ("-f", "--folder"):
            root_dir = arg
        elif opt in ("-e", "--ext"):
            ext = arg

    # We likely have to use the .flow() function of ImageDataGenerator to generate our own augmented data.
    # Keras does not provide sufficient documentation on the expected data format for the data format of the input array
    # but here's something:
    # - x is a Numpy array of rank 4 (num_images, image_width, image_height, channels).
    # - y is the corresponding labels. For greyscale image, channels must be equal to 1.

    # CREATE TRAINING DATA
    # Assumes a file name format like CA_001_HASHFORCARD.EXT - so the first 6 characters are used as labels.
    all_files = glob.glob(root_dir + '**/*' + ext, recursive=True)
    for idx, abs_filename in enumerate(all_files):
        foldername, filename = os.path.split(abs_filename)
        img_card = cv2.imread(abs_filename, cv2.IMREAD_COLOR)  # TODO: use cv2.IMREAD_UNCHANGED for alpha, needed for augmentation
        if 'x_train' not in locals():
            # It's A LOT faster to initialize x_train to a fixed size and then fill it up than to use np.append().
            x_train = np.empty((len(all_files), img_card.shape[0], img_card.shape[1], img_card.shape[2]))
            y_train = []
        x_train[idx, :, :, :] = img_card
        y_train += [filename[:6]]
        print_progress(idx+1, len(all_files), bar_length=20, prefix='Loading images: ', suffix='Done')

    aug = ImageDataGenerator(rescale=1./255,
                             rotation_range=15,
                             zoom_range=0.1,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             fill_mode="constant",
                             cval=1)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=20):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    Example:
        print_progress(i + 1, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
        where:
            - i is the index of the loop entry
            - l is the total number of entries in the loop
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    main(sys.argv[1:])

