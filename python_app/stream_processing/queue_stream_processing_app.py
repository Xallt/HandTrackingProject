import cv2
import logging
import colorlog
import multiprocessing
import ctypes
import time
from transformations.image_transform import ImageTransform
from .webcam_app import WebcamApp

class QueueStreamProcessingApp(WebcamApp):
    def __init__(
            self, 
            url: str, 
            transform: ImageTransform = None, 
            max_queue_size: int = 10,
            debug: bool = False
        ):
        super().__init__(url, debug)

        self.logger.info("Initializing QueueStreamProcessingApp")
        self.max_queue_size = max_queue_size
        self.transform = transform

        self.webcam_to_processing_queue = None
        self.processing_to_webcam_queue = None
        self.exit_flag = None
        self.webcam_lock = None

    def webcam_reader_loop(self):
        self.logger.info("Starting webcam reader loop")
        cap = cv2.VideoCapture(self.url)
        while True:
            if self.exit_flag.value:
                break
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to read frame, exiting")
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
        self.logger.info("Starting opencv window loop")
        prev_time = time.time()
        while True:
            if self.exit_flag.value:
                break
            frame = self.processing_to_webcam_queue.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            self.add_stats(
                frame,
                {'FPS': fps},
                color=(0, 255, 0),
                scale=0.8
            )
            cv2.imshow('frame', frame)
            prev_time = cur_time
            if cv2.waitKey(1) == ord('q'):
                self.logger.info("User pressed q, exiting")
                self.exit_flag.value = True
                break
        cv2.destroyAllWindows()

    def image_process_loop(self):
        self.logger.info("Starting image process loop")
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