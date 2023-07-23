import mediapipe as mp
from transformations.image_transform import ImageTransform

class MediaPipeHandDetectionTransform(ImageTransform):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
    def validate(self, image):
        return True
    def transform(self, image):
        image = image.copy()
        with self.mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
            results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image