import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from transformations.image_transform import ImageTransform
import pyautogui
import numpy as np
import cv2

SCREEN_SIZE = pyautogui.size()
HAND_LANDMARKER_PATH = 'assets/mediapipe/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# From https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    height, width, _ = rgb_image.shape
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            rgb_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(rgb_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

class MediaPipeHandDetectionTransform(ImageTransform):
    def __init__(self):
        self.sync_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
            running_mode=VisionRunningMode.IMAGE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def _draw_landmark_callback(self, result, image, timestamp):
        image = image.numpy_view()
        draw_landmarks_on_image(image, result)
        self.callback(image)

    def set_callback(self, callback):
        self.callback = callback
        self.options_async = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._draw_landmark_callback
        )

    def transform_async(self, image, timestamp):
        image = image.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        with HandLandmarker.create_from_options(self.options_async) as landmarker:
            landmarker.detect_async(mp_image, int(timestamp))
    def transform(self, image):
        image = image.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        with HandLandmarker.create_from_options(self.sync_options) as landmarker:
            results = landmarker.detect(mp_image)
        draw_landmarks_on_image(image, results)
        return image
