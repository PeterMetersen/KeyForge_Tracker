import sys, getopt
import tensorflow as tf


def main(argv):
    """
    Program to create a model.

    :param argv:
    :return:
    """

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

if __name__ == '__main__':
    main(sys.argv[1:])

