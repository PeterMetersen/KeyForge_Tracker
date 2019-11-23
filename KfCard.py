import os
import cv2
import numpy as np


class KfCard(object):
    def __init__(self, image, card_type):
        assert isinstance(image, np.ndarray), 'Input argument must be a numpy array.'
        self.image = image
        self.type = card_type

    @classmethod
    # Method to initialize from a file
    def from_file(cls, path, card_type, flag=cv2.IMREAD_COLOR):
        assert os.path.isfile(path), 'File not found.'
        np_image = cv2.imread(path, flag)
        return cls(image=np_image, type=card_type)

