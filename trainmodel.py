import sys, getopt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def main(argv):
    """
    Program to create a model.

    :param argv:
    :return:
    """

    folder_to_process = '/images'  # default folder

    try:
        opts, args = getopt.getopt(argv, "hf:", ["folder="])
    except getopt.GetoptError:
        print('Invalid input parameter.')
        print(main.__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(main.__doc__)
        elif opt in ("-f", "--folder"):
            folder_to_process = arg

    aug = ImageDataGenerator(rescale=1./255,
                             rotation_range=15,
                             zoom_range=0.1,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             fill_mode="constant",
                             cval=1)


if __name__ == '__main__':
    main(sys.argv[1:])

