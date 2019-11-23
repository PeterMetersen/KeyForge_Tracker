import cv2
import numpy as np
import os


def overlay_cards(folder, rectangle, thr, kernel=(3, 3), display_image=False):
    """
    Reads in all image files in folder and overlays the defined area. Pixels that are darker than the defined threshold
    when converted to gray-scale will be neglected in the overlay by considering them as transparent. This is used to
    discard words on the card.

    :param folder:  Folder to parse
    :param rectangle:  Tuple [y,x,h,w]. Defines what region of the cards to look at.
    :param thr:  Thresholding value. Defines which regions to neglect when overlaying images.
    :param kernel:  Kernel size for applying dilation.
    :param display_image:  Displays images during execution. Used for debugging.

    :return:  Numpy array of overlayed card. Without alpha channel.
    """

    extension = '.png'
    y, x, h, w = rectangle
    files = [f for f in os.listdir(folder) if f.endswith(extension)]
    # Create baseline overlay image
    overlayed_img = np.zeros((h, w, 3))
    for cfile in files:
        #row, column = img.shape
        #[0,0] - top left
        #BGR
        img = cv2.imread(os.path.join(folder, cfile))  # read in as colored, without transparency
        img = img[y:y+h, x:x+w]
        # Create alpha mask: the letters and their neighbourhood shall be considered as "transparent" when overlaying.
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha_mask = cv2.threshold(bw_img, thr, 255, cv2.THRESH_BINARY)[1]  # 0: do not consider, 1: do consider
        alpha_mask = cv2.erode(alpha_mask, np.ones(kernel, np.uint8), 1)  # increase the black area
        if display_image:
            cv2.imshow('mask', alpha_mask)
            cv2.waitKey(0)
        alpha_mask = np.stack((alpha_mask, alpha_mask, alpha_mask), axis=2)

        # Overlay current image on the accumulated overlay image
        overlayed_img = alpha_blending(img, overlayed_img, alpha_mask)

    return overlayed_img


def repaint_missing(img, thr, radius=3, display_image=False):
    """
    This method can be used to interpolate missing parts after cards were overlayed.

    :param img:  Numpy array with 3 channels.
    :param thr:  Thresholding value. Defines which regions to consider as missing part.
    :param radius:  Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
    :param display_image:  Display image after interpolation or not.
    :return:  Numpy array with the corrected image.
    """

    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    missing_mask = cv2.threshold(bw_img, thr, 255, cv2.THRESH_BINARY_INV)[1]  # non-0 elements define the missing part
    retouched_img = cv2.inpaint(img, missing_mask, radius, cv2.INPAINT_TELEA)
    if display_image:
        cv2.imshow('Inpainted image', retouched_img)
        cv2.waitKey(0)
    return retouched_img


def alpha_blending(foreground, background, alpha):
    """
    Function to add two images with alpha mask. Images must be uint8 without alpha channel.

    :param foreground:  uint8 numpy array with 3 channels, i.e. without alpha channel
    :param background:  uint8 numpy array with 3 channels, i.e. without alpha channel
    :param alpha:  uint8 numpy array with 3 channels, i.e. without alpha channel
    :return:  blended image as uint8 numpy array with 3 channels, i.e. without alpha channel
    """
    display_image = False

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground/background with the alpha matte and add them
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(np.ones(alpha.shape) - alpha, background)
    blended_img = cv2.add(foreground, background)

    if display_image: # Display image
        cv2.imshow("Alpha blended images", blended_img / 255)
        cv2.waitKey(0)

    return blended_img.astype(np.uint8)


if __name__ == '__main__':
    # Assuming following folder structure:
    #    parent_folder
    #      \ brobnar
    #          \ action
    #          \ artifact
    #          \ creature
    #          \ upgrade
    #      \ dis
    #          \ ...
    parent_folder = "C:/Users/Titan/Documents/Downloads/all_by_house/"

    # Card-type dependent overlay areas
    overlay_areas = {
        'artifact': [(260, 70, 20, 160),
                     (280, 25, 100, 240)],
        'action':   [(280, 25, 98, 242)],
        'creature': [(270, 50, 19, 198),
                     (286, 28, 92, 239)],
        'upgrade':  [(66, 39, 102, 227)]
    }

    # House-dependent thresholds for 3x3 kernels
    kernel_size = (3,3)
    thresholds = {
        'brobnar': 180,
        'dis': 170, # 160
        'logos': 200,
        'mars': 170,
        'sanctum': 190,
        'shadows': 160,
        'untamed': 160
    }

    # # House-dependent thresholds with 4x4 kernels
    # kernel_size = (4,4)
    # thresholds = {
    #     'brobnar': 140,
    #     'dis': 130,
    #     'logos': 150,
    #     'mars': 120,
    #     'sanctum': 140,
    #     'shadows': 140,
    #     'untamed': 140
    # }

    for house in thresholds.keys():
        for card_type in overlay_areas.keys():
            c_folder = os.path.join(parent_folder, house, card_type)
            files = [f for f in os.listdir(c_folder) if f.endswith('.png')]
            base_file = cv2.imread(os.path.join(c_folder, files[-1]))
            for c_area in overlay_areas[card_type]:
                y, x, h, w = c_area
                overlay_img = overlay_cards(c_folder, c_area, thresholds[house], kernel_size, False)
                # INTERPOLATE HERE IF NEEDED
                overlay_img = repaint_missing(overlay_img, 128)
                base_file[y:y+h,x:x+w] = overlay_img
                if False: # card_type in ['creature','upgrade']:
                    cv2.imshow('test img', base_file)
                    cv2.waitKey()
            cv2.imwrite(os.path.join(parent_folder, house, card_type, 'output.png'), base_file)
    # get all unique pixels: set( tuple(v) for m2d in img for v in m2d )
    #   for m2d in img: loops through img on its first dimension. E.g.: if img is 100x200x3: returns 200x3 100 times
    #   for v in m2d: for each result in previous call loops through 1st dimension. E.g.: if 200x3 --> 3 200 times
