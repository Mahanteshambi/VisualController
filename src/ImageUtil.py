import cv2
import numpy as np

class ImageUtils:
    def __init__(self) -> None:
        self.zoom_scale = 0

    def zoom_image(self, image, scale):
        height, width, channels = image.shape
        centerX,centerY=int(height/2),int(width/2)
        self.zoom_scale += scale
        # print('zoom_scale: ', self.zoom_scale)
        self.zoom_scale = max(0, self.zoom_scale)
        self.zoom_scale = min(49, self.zoom_scale)
        radiusX,radiusY= int(self.zoom_scale*height/100),int(self.zoom_scale*width/100)
        radiusX = int((height - (2 * self.zoom_scale * height / 100))/2)
        radiusY = int((width - (2 * self.zoom_scale * width / 100))/2)
        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY

        cropped = image[minX:maxX, minY:maxY]
        resized_cropped = cv2.resize(cropped, (width, height)) 
        return resized_cropped

    def rotate_image(self, image, degrees):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, degrees, 1.0)
        result = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result