import os
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import mediapipe as mp

class ImageUtils:
    def __init__(self, mp_hands):
        self.zoom_scale = 0
        self.mp_hands = mp_hands
        self.drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing = mp.solutions.drawing_utils
        zoom_img_path =  os.path.join(os.getcwd(), 'imgs/zoom-b.png')
        draw_img_path =  os.path.join(os.getcwd(), 'imgs/draw-b.png')
        rotate_img_path =  os.path.join(os.getcwd(), 'imgs/rotate-b.png')
        self.zoom_img = Image.open(zoom_img_path)
        self.zoom_img = self.zoom_img.resize((100, 50), Image.BICUBIC)
        self.zoom_img.convert('RGB')
        self.draw_img = Image.open(draw_img_path)
        self.draw_img = self.draw_img.resize((100, 50), Image.BICUBIC)
        self.rotate_img = Image.open(rotate_img_path)
        self.rotate_img = self.rotate_img.resize((100, 50), Image.BICUBIC)


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
        height, width, channels = image.shape
        self.mp_drawing.draw_landmarks(
            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.drawing_styles.get_default_hand_landmark_style(),
            self.drawing_styles.get_default_hand_connection_style())
        xlist = [hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].x]
        ylist = [hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y]
        minx = min(xlist) * width
        maxx = max(xlist) * width
        miny = min(ylist) * height
        maxy = max(ylist) * height
        cv2.rectangle(image, (int(minx), int(miny)), (int(maxx), int(maxy)), (255, 255, 255), 5)
        
    def blend_magic_circle(self, image, hand_landmarks):
        height, width, channels = image.shape
        xlist = [hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].x,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].x]
        ylist = [hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y]
        minx = min(xlist) * width
        maxx = max(xlist) * width
        miny = min(ylist) * height
        maxy = max(ylist) * height
        pil_image = Image.fromarray(image)
        newsize = (int(maxx-minx), int(maxy-miny))
        self.blend_img_pil = self.blend_img_pil.resize(newsize)
        pil_image.paste(self.blend_img_pil, (int(minx), int(miny)))
        return np.asarray(pil_image)

    def draw_hand_reference(self, drawable_img, start_xy, end_xy):
        cv2.circle(drawable_img, start_xy, 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(drawable_img, end_xy, 10, (0, 255, 255), cv2.FILLED)
        cv2.line(drawable_img, start_xy, end_xy, (255, 255, 255), 3)

    def draw_on_screen(self, image, drawable_xy):
        for (x, y) in drawable_xy:
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), cv2.FILLED)

    def add_indicators(self, image):
        pil_image = Image.fromarray(image)
        pil_image.paste(self.zoom_img, (10, 10))
        pil_image.paste(self.draw_img, (10, 420))
        pil_image.paste(self.rotate_img, (360, 10))
        draw = ImageDraw.Draw(pil_image)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        draw.text((110, 30),"Zoom Controls",(255,255,255))
        draw.text((110, 450),"Draw Controls",(255,255,255))
        draw.text((460, 30),"rotate Controls",(255,255,255))
        return np.asarray(pil_image)
