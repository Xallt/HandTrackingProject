import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from transformations.image_transform import ImageTransform
import pyautogui
import numpy as np
import cv2
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class MediaPipeHandDetectionTransform(ImageTransform):
    def __init__(self):
        self.hands = None

    def set_callback(self, callback):
        self.callback = callback

    def transform_async(self, image, timestamp):
        image = self.transform(image)
        self.callback(image)
    def transform(self, image):
        # Initialize the solution class in the first call to transform
        # Otherwise it will be copied across processes and that breaks the solution
        if self.hands is None:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands = 2
            )


        image = image.copy()
        hands_result = self.hands.process(image)
        if hands_result.multi_hand_landmarks is None:
            return image
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        return image

