#Preprocessors
import cv2
import numpy as np
import imutils
class AspectAwarePreprocess:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dH, dW = 0, 0
        if w <= h:
            image = imutils.resize(image, width = self.width, inter = self.inter)
            dH = (image.shape[0] - self.height)//2
        else:
            image = imutils.resize(image, height = self.height, inter = self.inter)
            dW = (image.shape[1] - self.width)//2
        (h, w) = image.shape[:2]
        image = image[dH:h-dH, dW:w-dW]
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)
