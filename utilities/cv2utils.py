import cv2
import numpy as np

class CV2_HELPER:

    def __init__(self):
        pass
    
    # Returns a binary image using an adaptative threshold
    def binarization_adaptative_threshold(self, image):
        # 11 => size of a pixel neighborhood that is used to calculate a threshold value for the pixel
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def binarization_otsu(self, image):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    # smoothen the image by removing small dots/patches which have high intensity than the rest of the image
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

