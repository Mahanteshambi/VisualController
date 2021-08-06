import cv2
import os
import mediapipe as mp
import math
from google.protobuf.json_format import MessageToDict
from Gesture_util import *
import numpy as np
from Gesture_util import GestureUtil
from ImageUtil import ImageUtils

class HandController:

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.gesture_utils = GestureUtil(self.mp_hands)
        self.image_width, self.image_height = 0, 0
        self.curr_factor = -100
        self.rotate_factor = -100.0
        self.is_editable = False
        self.image_utils = ImageUtils(self.mp_hands)
        self.drawable_xy = list()
        self.drawable_y = list()

    def start_reading_cam(self):
        cap = cv2.VideoCapture(0)
        with self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.image_height, self.image_width, _ = image.shape
                drawable_img = np.zeros(image.shape, np.uint8)
                 
                left_hand_gesture = HandGesture.OPEN
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        which_hand = results.multi_handedness[idx].classification[0].label
                        if which_hand == 'Left':
                            left_hand_gesture = self.gesture_utils.determine_gesture(hand_landmarks)
                            self.image_utils.draw_hand_landmarks(drawable_img, hand_landmarks)
                            which_hand = ''
                        elif which_hand == 'Right':
                            if left_hand_gesture:
                                image, drawable_img = self.perform_right_hand_operation(self.mp_hands, hand_landmarks, 
                                image, left_hand_gesture, drawable_img)
                            which_hand = ''
                
                image = cv2.hconcat([image, drawable_img])
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    

    def perform_right_hand_operation(self, mp_hands, hand_landmarks, image, left_hand_gesture, drawable_img):
        self.image_height, self.image_width, _ = image.shape
        x1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * self.image_width
        y1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * self.image_height
        x2 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.image_width
        y2 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.image_height
        length = math.hypot(x2-x1, y2-y1)
        factor = length / 5
        self.is_editable = left_hand_gesture == HandGesture.CLOSE
        if self.is_editable:
            right_hand_gesture = self.gesture_utils.determine_gesture(hand_landmarks=hand_landmarks)
            print('right hand gesture', right_hand_gesture)
            if right_hand_gesture == HandGesture.ZOOM:
                curr_length = length
                if self.curr_factor != -100:
                    image = self.image_utils.zoom_image(image, self.curr_factor - factor)
                    start_xy = (int(x1), int(y1))
                    end_xy = (int(x2), int(y2))
                    self.image_utils.draw_hand_reference(drawable_img, (int(x1), int(y1)), (int(x2), int(y2)))
                self.curr_factor = factor
            elif right_hand_gesture == HandGesture.ROTATE:
                degrees = self.gesture_utils.get_angle(hand_landmarks)
                if self.rotate_factor == -100.0:
                    self.rotate_factor = degrees
                else:
                    image = self.image_utils.rotate_image(image, int(self.rotate_factor) - int(degrees))
                    self.image_utils.draw_hand_reference(drawable_img, (int(x1), int(y1)), (int(x2), int(y2)))
            elif right_hand_gesture == HandGesture.DRAW:
                x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.image_width
                y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.image_height
                self.drawable_xy.append((int(x), int(y)))
                self.image_utils.draw_on_screen(image, self.drawable_xy)
                
        self.image_utils.draw_hand_landmarks(drawable_img, hand_landmarks)
        return image, drawable_img



if __name__ == '__main__':
    hand_controller = HandController()
    hand_controller.start_reading_cam()