import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#  Global Variables
DOWN_SCALE = 0.3



def main(argv):

    # Load the source image
    imageName ='4Cards.jpg'

    src = cv.imread(cv.samples.findFile(imageName))

    print('Size of pic: ', np.shape(src),' - Scaling dimensions by a factor: ', DOWN_SCALE)
    res = cv.resize(src, None, fx=DOWN_SCALE, fy=DOWN_SCALE, interpolation=cv.INTER_CUBIC)
    print('Size of pic after scaling: ', np.shape(res))
    src = res
    #cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    #cv.imshow('1', src)
    edges = cv.Canny(src, 100, 200)
    edges= cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    numpy_horizontal = np.hstack((src, edges))
    cv.imshow('2', numpy_horizontal)
    cv.waitKey(0)
    cv.imwrite('EdgeDetection.jpg',numpy_horizontal)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
