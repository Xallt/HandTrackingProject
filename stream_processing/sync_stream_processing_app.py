import cv2
import logging
import colorlog
import multiprocessing
import ctypes
import time
import numpy as np
from transformations.image_transform import ImageTransform
from .webcam_app import WebcamApp

class SynchronousStreamProcessingApp(WebcamApp):
    def __init__(
            self, 
            url: str, 
            transform: ImageTransform = None, 
            debug: bool = False
        ):
        super().__init__(url, debug)

        self.logger.info("Initializing SynchronousStreamProcessingApp")
        self.transform = transform

    def run(self):
        cap = cv2.VideoCapture(self.url)
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to read frame")
                break

            ##
            # Frame processing
            ##

            # Start timer
            start_frame_time = time.time()
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip frame so it isn't mirrored
            frame = cv2.flip(frame, 1)
            # Apply transform
            if self.transform is not None:
                frame = self.transform.transform(frame)
            # Convert back to BGR for OpenCV's imshow
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Calculate time to process frame
            sec_per_frame = time.time() - start_frame_time

            ##
            # Display
            ##

            # Calculate FPS
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            prev_time = cur_time

            # Display FPS
            self.add_stats(
                frame,
                {'FPS': fps, 'sec/frame': sec_per_frame},
                color=(0, 255, 0),
                scale=0.8
            )

            cv2.imshow("frame", frame)

            # Exit on q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("User pressed q, exiting")
                break
        cap.release()
        cv2.destroyAllWindows()

