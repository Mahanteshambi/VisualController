import cv2
import mediapipe as mp
import math
from google.protobuf.json_format import MessageToDict

class HandController:

    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.drawing_styles = mp.solutions.drawing_styles
        self.image_width, self.image_height = 0, 0
        self.zoom_scale = 1
        self.curr_factor = -100
        self.is_editable = False
        self.zoom_scale = 0

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
                image_height, image_width, _ = image.shape
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        which_hand = results.multi_handedness[idx].classification[0].label
                        # print(results.multi_handedness[idx].classification)
                        print('Hand is ' + which_hand)
                        if which_hand == 'Left':
                            self.determine_gesture(hand_landmarks)
                            # draw_hand(mp_hands, hand_landmarks, image)
                            which_hand = ''
                        elif which_hand == 'Right':
                            image = self.perform_right_hand_operation(self.mp_hands, hand_landmarks, image)
                            which_hand = ''

                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    def determine_gesture(self, hand_landmarks):

        is_thumb_closed = self.is_finger_closed(
                hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].x,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x
        )

        is_index_finger_closed = self.is_finger_closed(
                hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        )

        is_middle_finger_closed = self.is_finger_closed(
                hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)

        is_ring_finger_closed = self.is_finger_closed(
                hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y
        )

        is_pinky_finger_closed = self.is_finger_closed(
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y, 
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y,
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y)

        if is_thumb_closed and is_index_finger_closed and \
            is_middle_finger_closed and is_ring_finger_closed and\
            is_pinky_finger_closed:
            print('Hand is closed')
            self.is_editable = True
        else:
            print('Hand is open')
            self.is_editable = False

    def is_finger_closed(self, pip, dip, tip):
        if dip < pip and tip < dip:
            return False
        elif pip < dip and dip < tip:
            return True
        else:
            return True

    def perform_right_hand_operation(self, mp_hands, hand_landmarks, image):
        self.image_height, self.image_width, _ = image.shape
        x1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * self.image_width#hand_landmarks[4][1], hand_landmarks[4][2]
        y1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * self.image_height
        x2 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.image_width#hand_landmarks[8][1], hand_landmarks[8][2]
        y2 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.image_height
        length = math.hypot(x2-x1, y2-y1)
        factor = length / 10
        
        print(factor, 'iseditable: ', self.is_editable)
        # print(image.shape)
        # image = scale_image(image, factor=factor)
        if self.is_editable:
            curr_length = length
            print('currfactor: ', self.curr_factor)
            if self.curr_factor != -100:
                image = self.zoom_image(image, self.curr_factor - factor)
            self.curr_factor = factor
        # print(image.shape)
        cv2.circle(image, (int(x1), int(y1)), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (int(x2), int(y2)), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
        self.mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            self.drawing_styles.get_default_hand_landmark_style(),
            self.drawing_styles.get_default_hand_connection_style())
        return image

    
    def zoom_image(self, image, scale):
        height, width, channels = image.shape
        centerX,centerY=int(height/2),int(width/2)
        global zoom_scale
        self.zoom_scale += scale
        print('zoom_scale: ', self.zoom_scale)
        self.zoom_scale = max(0, self.zoom_scale)
        self.zoom_scale = min(49, self.zoom_scale)
        radiusX,radiusY= int(self.zoom_scale*height/100),int(self.zoom_scale*width/100)
        radiusX = int((height - (2 * self.zoom_scale * height / 100))/2)
        radiusY = int((width - (2 * self.zoom_scale * width / 100))/2)
        # print('center', centerX, centerY)
        # print('radius: ', radiusX, radiusY)
        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY
        # print('min, max', minX, minY, maxX, maxY)

        cropped = image[minX:maxX, minY:maxY]
        resized_cropped = cv2.resize(cropped, (width, height)) 
        return resized_cropped

if __name__ == '__main__':
    hand_controller = HandController()
    hand_controller.start_reading_cam()