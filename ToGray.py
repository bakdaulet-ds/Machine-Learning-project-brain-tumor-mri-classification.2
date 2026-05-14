#Preprocessors
import cv2
import numpy as np
import imutils
class ToGray:
    @staticmethod
    def preprocess(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
