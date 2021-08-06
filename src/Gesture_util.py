import math
from enum import Enum
from math import degrees

class HandGesture(Enum):
    OPEN = 1
    CLOSE = 2
    DRAW = 3
    ZOOM = 4
    ROTATE = 5
    FOUR = 6
    FIVE = 7

class GestureUtil():

    def __init__(self, mp_hands):
        self.mp_hands = mp_hands

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
            # print('Hand is closed')
            return HandGesture.CLOSE
        elif not is_thumb_closed and not is_index_finger_closed and \
            is_middle_finger_closed and is_ring_finger_closed and is_pinky_finger_closed:
            return HandGesture.ZOOM
        elif not is_thumb_closed and not is_index_finger_closed and not is_middle_finger_closed and\
            is_ring_finger_closed and is_pinky_finger_closed:
            return HandGesture.ROTATE
        elif is_thumb_closed and is_middle_finger_closed and is_ring_finger_closed and is_pinky_finger_closed\
            and not is_index_finger_closed:
            return HandGesture.DRAW
        else:
            # print('Hand is open')
            return HandGesture.OPEN

    def get_angle(self, hand_landmarks):
        x1 = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x
        y1 = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y
        x2 = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
        y2 = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

        radians = math.atan2(y1 - y2, x1 - x2)
        degrees = math.degrees(radians)

        return degrees

    def is_finger_closed(self, pip, dip, tip):
        if dip < pip and tip < dip:
            return False
        elif pip < dip and dip < tip:
            return True
        else:
            return True