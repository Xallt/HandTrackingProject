import cv2
import logging
import colorlog
import multiprocessing
import ctypes
import time
import numpy as np
from transformations.image_transform import ImageTransform

class SynchronousStreamProcessingApp:
    def __init__(
            self, 
            url: str, 
            transform: ImageTransform = None, 
            debug: bool = False
        ):
        self._init_logging(debug)

        self.logger.info("Initializing SynchronousStreamProcessingApp")
        self.url = self._preprocess_url(url)
        self.transform = transform

    def _init_logging(self, debug: bool):
        self.logger = logging.getLogger()
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        if debug:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)

        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def _preprocess_url(self, url: str) -> str:
        if url.isnumeric():
            url = int(url)
        return url

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
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"sec/frame: {sec_per_frame:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("frame", frame)

            # Exit on q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("User pressed q, exiting")
                break
        cap.release()
        cv2.destroyAllWindows()

