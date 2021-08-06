import cv2
import numpy as np
import mediapipe as mp

class ImageUtils:
    def __init__(self, mp_hands):
        self.zoom_scale = 0
        self.mp_hands = mp_hands
        self.drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing = mp.solutions.drawing_utils

    def zoom_image(self, image, scale):
        height, width, channels = image.shape
        centerX,centerY=int(height/2),int(width/2)
        self.zoom_scale += scale
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

    def draw_hand_landmarks(self, image, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.drawing_styles.get_default_hand_landmark_style(),
            self.drawing_styles.get_default_hand_connection_style())

    def draw_hand_reference(self, drawable_img, start_xy, end_xy):
        cv2.circle(drawable_img, start_xy, 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(drawable_img, end_xy, 10, (0, 255, 255), cv2.FILLED)
        cv2.line(drawable_img, start_xy, end_xy, (255, 255, 255), 3)

    def draw_on_screen(self, image, drawable_xy):
        for (x, y) in drawable_xy:
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), cv2.FILLED)