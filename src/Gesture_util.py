import math
from enum import Enum
from math import degrees


class HandGesture(Enum):
    OPEN = 1
    CLOSE = 2
    DRAW = 3
    ZOOM = 4
    ROTATE = 5


class GestureUtil:
    """Utility class for detecting hand gestures using mediapipe hand landmarks."""

    def __init__(self, mp_hands):
        self.mp_hands = mp_hands

    def determine_gesture(self, hand_landmarks):
        """Method to determine the position of fingers irrespective of left or right hand. Details of points is available at
            https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
        Args:
            hand_landmarks ((x,y) of all 21 hand landmarks): Hand landmark points position in terms of 2d coordinates

        Returns:
            HandGesture: If hand is open, close, or any other gesture
        """
        is_thumb_closed = self.is_finger_closed(
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
        )

        is_index_finger_closed = self.is_finger_closed(
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
        )

        is_middle_finger_closed = self.is_finger_closed(
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
        )

        is_ring_finger_closed = self.is_finger_closed(
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y,
        )

        is_pinky_finger_closed = self.is_finger_closed(
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y,
        )
        print()
        if (
            is_thumb_closed
            and is_index_finger_closed
            and is_middle_finger_closed
            and is_ring_finger_closed
            and is_pinky_finger_closed
        ):
            return HandGesture.CLOSE
        elif (
            not is_thumb_closed
            and not is_index_finger_closed
            and is_middle_finger_closed
            and is_ring_finger_closed
            and is_pinky_finger_closed
        ):
            return HandGesture.ZOOM
        elif (
            not is_thumb_closed
            and not is_index_finger_closed
            and not is_middle_finger_closed
            and is_ring_finger_closed
            and is_pinky_finger_closed
        ):
            return HandGesture.ROTATE
        elif (
            is_thumb_closed
            and is_middle_finger_closed
            and is_ring_finger_closed
            and is_pinky_finger_closed
            and not is_index_finger_closed
        ):
            return HandGesture.DRAW
        else:
            return HandGesture.OPEN

    def get_angle(self, hand_landmarks):
        """Method to find angle of rotation w.r.t thumb and middle finger

        Args:
            hand_landmarks (Mediapipe hand landmarks): Mediapipe hand landmarks

        Returns:
            Degree of rotation: Degree
        """
        x1 = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x
        y1 = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y
        x2 = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
        y2 = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

        radians = math.atan2(y1 - y2, x1 - x2)
        degrees = math.degrees(radians)

        return degrees

    def is_finger_closed(self, pip, dip, tip):
        """Method to determine if finger is closed or open

        Returns:
            boolean: True if finger is closed, else false
        """
        if dip < pip and tip < dip:
            return False
        elif pip < dip and dip < tip:
            return True
        else:
            return True
