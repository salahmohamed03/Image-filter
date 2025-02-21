
import numpy as np
from PIL import Image as PILImage
from PyQt6.QtGui import QPixmap,QImage
from functools import lru_cache
import cv2
class Image:
    def __init__(self, path = None, image_data = None):
        if path is None:
            self.path = None
            self.qimg = QImage(image_data, image_data.shape[1], image_data.shape[0], QImage.Format.Format_RGB888)
            self.image_data = image_data
        else:
            self.path = path
            self.qimg, self.image_data = Image.load_image(path)

    def resize(self, new_size):
        self.image_data.resize(new_size,refcheck=False)
        height, width, channel = self.image_data.shape
        bytes_per_line = 3 * width
        self.qimg = QImage(self.image_data,width, height, bytes_per_line, QImage.Format.Format_RGB888)
   
    @staticmethod
    def load_image(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg, img
    
    
@lru_cache(maxsize=1)
class Images:
    def __init__(self):
        self.image1:Image = None
        self.image2:Image = None
        self.output1:Image = None
        self.output2:Image = None