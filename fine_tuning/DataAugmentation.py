#Fine-tuning
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
class DataAugmentation:
    def __init__(self, angle=15, width=0.1, height=0.1, shear=0.1, zoom=0.1, h_flip=False):
        self.angle = angle
        self.width = width
        self.height = height
        self.shear = shear
        self.zoom = zoom
        self.h_flip = h_flip
    def imageGenerator(self, X, y, batch_size = 32):
        aug = ImageDataGenerator(rotation_range = self.angle,
                                width_shift_range = self.width,
                                height_shift_range = self.height,
                                shear_range = self.shear,
                                zoom_range = self.zoom,
                                horizontal_flip = self.h_flip
                                )
        imageGen = aug.flow(X, y, batch_size = batch_size)
        return imageGen
