import cv2
import logging
import colorlog
import multiprocessing
import ctypes
import time
import numpy as np
from transformations.image_transform import ImageTransform
from multiprocessing.shared_memory import SharedMemory

class SharedmemStreamProcessingApp:
    def __init__(
            self, 
            url: str, 
            transform: ImageTransform = None, 
            debug: bool = False
        ):
        self._init_logging(debug)

        self.logger.info("Initializing StreamProcessingApp")
        self.url = self._preprocess_url(url)
        self.transform = transform

        self.webcam_to_processing_queue = None
        self.processing_to_webcam_queue = None
        self.exit_flag = None
        self.shm_event = None
        self.shared_mem_lock = None

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

    def webcam_reader_loop(self):
        self.logger.info("Starting webcam reader loop")
        cap = cv2.VideoCapture(self.url)
        # Get the width and height of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        ##
        # Shared memory approach from
        # https://stackoverflow.com/questions/66353952/how-to-pass-video-stream-from-one-python-script-to-another/66522825#66522825
        ##

        frame_buffer_shm = SharedMemory(name='cur_frame', create=True, size=width * height * 3)
        frame = np.ndarray((height, width, 3), buffer=frame_buffer_shm.buf, dtype='u1')
        frame_shape_shm = SharedMemory(name="frame_shape", create=True, size=3*4) #4 bytes per dim as long as int32 is big enough
        self.shm_event.set()
        frame_shape = np.ndarray(3, buffer=frame_shape_shm.buf, dtype='i4')  #4 bytes per dim as long as int32 is big enough
        frame_shape[:] = np.array([height, width, 3])

        while True:
            if self.exit_flag.value:
                break

            ret, a = cap.read(frame)
            frame[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        self.webcam_to_processing_queue.cancel_join_thread()
        self.processing_to_webcam_queue.cancel_join_thread()
        frame_shape_shm.close()
        frame_buffer_shm.close()
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
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            prev_time = cur_time
            if cv2.waitKey(1) == ord('q'):
                self.logger.info("User pressed q, exiting")
                self.exit_flag.value = True
                break
        cv2.destroyAllWindows()

    def image_process_loop(self):
        self.logger.info("Starting image process loop")
        self.shm_event.wait()
        frame_shape_shm = SharedMemory(name="frame_shape")
        frame_shape = np.ndarray(3, buffer=frame_shape_shm.buf, dtype='i4')
        frame_shm = SharedMemory(name='cur_frame')
        frame = np.ndarray(frame_shape, buffer=frame_shm.buf, dtype='u1')
        while True:
            if self.exit_flag.value:
                break
            frame_res = cv2.flip(frame, 1)
            frame_res = self.transform.transform(frame_res)
            self.processing_to_webcam_queue.put(frame_res)
        self.webcam_to_processing_queue.cancel_join_thread()
        self.processing_to_webcam_queue.cancel_join_thread()
        frame_shape_shm.close()
        frame_shm.close()

    def run(self):
        self.webcam_to_processing_queue = multiprocessing.Queue()
        self.processing_to_webcam_queue = multiprocessing.Queue()
        self.exit_flag = multiprocessing.Value(ctypes.c_bool, False)
        self.shm_event = multiprocessing.Event()
        self.shared_mem_lock = multiprocessing.Lock()

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