import cv2
import argparse
import numpy as np
import time
import logging
import colorlog
import multiprocessing
import ctypes

from transformations.mp_hand_detection import ImageTransform, MediaPipeHandDetectionTransform


class StreamProcessingApp:
    def __init__(self, url: str, transform: ImageTransform = None, max_queue_size: int = 10):
        self._init_logging()

        self.logger.info("Initializing StreamProcessingApp")
        self.url = self._preprocess_url(url)
        self.max_queue_size = max_queue_size
        self.transform = transform

        self.webcam_to_processing_queue = None
        self.processing_to_webcam_queue = None
        self.exit_flag = None
        self.webcam_lock = None

    def _init_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

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

    def webcam_reader_loop(self):
        cap = cv2.VideoCapture(self.url)
        while True:
            if self.exit_flag.value:
                break
            ret, frame = cap.read()
            if not ret:
                self.exit_flag.value = True
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.webcam_lock:
                if self.webcam_to_processing_queue.qsize() == self.max_queue_size:
                    self.webcam_to_processing_queue.get()
                self.webcam_to_processing_queue.put(frame)
        cap.release()
        self.webcam_to_processing_queue.cancel_join_thread()
        self.processing_to_webcam_queue.cancel_join_thread()
    def opencv_window_loop(self):
        prev_time = time.time()
        while True:
            if self.exit_flag.value:
                break
            frame = self.processing_to_webcam_queue.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            prev_time = cur_time
            if cv2.waitKey(1) == ord('q'):
                self.exit_flag.value = True
                break
        cv2.destroyAllWindows()

    def image_process_loop(self):
        while True:
            if self.exit_flag.value:
                break
            frame = self.webcam_to_processing_queue.get()
            with self.webcam_lock:
                while not self.webcam_to_processing_queue.empty():
                    frame = self.webcam_to_processing_queue.get()
            frame = cv2.flip(frame, 1)
            frame = self.transform.transform(frame)
            self.processing_to_webcam_queue.put(frame)
        self.webcam_to_processing_queue.cancel_join_thread()
        self.processing_to_webcam_queue.cancel_join_thread()

    def run(self):
        self.webcam_to_processing_queue = multiprocessing.Queue()
        self.processing_to_webcam_queue = multiprocessing.Queue()
        self.exit_flag = multiprocessing.Value(ctypes.c_bool, False)
        self.webcam_lock = multiprocessing.Lock()

        transformation_process = multiprocessing.Process(
            target=self.image_process_loop,
            args=()
        )
        webcam_reader_process = multiprocessing.Process(
            target=self.webcam_reader_loop,
            args=()
        )

        webcam_reader_process.start()
        transformation_process.start()
        self.opencv_window_loop()
        transformation_process.join()
        webcam_reader_process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live processing of video from phone')

    parser.add_argument('url', type=str, help='URL of video stream')

    args = parser.parse_args()

    app = StreamProcessingApp(args.url, MediaPipeHandDetectionTransform())
    app.run()

