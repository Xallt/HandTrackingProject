import cv2
import logging
import colorlog
import multiprocessing
import ctypes
import time
import numpy as np
from transformations.image_transform import ImageTransform
from multiprocessing.shared_memory import SharedMemory
from .webcam_app import WebcamApp

class SharedmemStreamProcessingApp(WebcamApp):
    def __init__(
            self, 
            url: str, 
            transform: ImageTransform = None, 
            debug: bool = False,
            async_transform: bool = False
        ):
        super().__init__(url, debug)

        self.logger.info("Initializing SharedmemStreamProcessingApp")
        self.transform = transform
        self.async_transform = async_transform

        self.frame_counter = None
        self.resolution_set_event = None
        self.frame_buffer_shm = None
        self.frame_shape_shm = None

        self.exit_flag = None
        self.shared_mem_lock = None

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

        frame_shape_shm = SharedMemory(name="frame_shape")
        frame_shape = np.ndarray(2, buffer=frame_shape_shm.buf, dtype='i4')
        frame_shape[:] = np.array([height, width])
        self.resolution_set_event.set()
        self.frame_buffer_init_event.wait()
        frame_shm = SharedMemory(name='cur_frame')
        frame = np.ndarray((*frame_shape, 3), buffer=frame_shm.buf, dtype='u1')

        while True:
            if self.exit_flag.value:
                break

            ret, _ = cap.read(frame)
        cap.release()
        frame_shape_shm.close()
        frame_shm.close()
    def opencv_window_loop(self):
        self.logger.info("Starting opencv window loop")
        frame_shape_shm = SharedMemory(name="frame_shape")
        frame_shape = np.ndarray(2, buffer=frame_shape_shm.buf, dtype='i4')
        frame_shm = SharedMemory(name='cur_processed_frame')
        processed_frame = np.ndarray((*frame_shape, 3), buffer=frame_shm.buf, dtype='u1')
        prev_time = time.time()
        last_counter_value = self.frame_counter.value
        while True:
            if self.exit_flag.value:
                break
            if last_counter_value == self.frame_counter.value:
                continue
            last_counter_value = self.frame_counter.value
            frame_res = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            self.add_stats(
                frame_res,
                {'FPS': fps},
                color=(0, 255, 0),
                scale=0.8
            )
            cv2.imshow('frame', frame_res)
            prev_time = cur_time
            if cv2.waitKey(1) == ord('q'):
                self.logger.info("User pressed q, exiting")
                self.exit_flag.value = True
                break
        cv2.destroyAllWindows()
        frame_shm.close()

    def image_process_loop(self):
        self.start_time = time.time() * 1000

        self.logger.info("Starting image process loop")
        self.frame_buffer_init_event.wait()
        frame_shape_shm = SharedMemory(name="frame_shape")
        frame_shape = np.ndarray(2, buffer=frame_shape_shm.buf, dtype='i4')
        frame_shm = SharedMemory(name='cur_frame')
        frame = np.ndarray((*frame_shape, 3), buffer=frame_shm.buf, dtype='u1')
        processed_frame_shm = SharedMemory(name='cur_processed_frame')
        processed_frame = np.ndarray((*frame_shape, 3), buffer=processed_frame_shm.buf, dtype='u1')
        def register_transformed_image(image: np.ndarray):
            self.logger.debug(f"Received transformed frame {self.frame_counter.value}")
            processed_frame[:] = image
            self.frame_counter.value += 1

        if self.async_transform:
            self.transform.set_callback(register_transformed_image)
        while True:
            if self.exit_flag.value:
                break
            frame_res = cv2.flip(frame, 1)
            frame_res = cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB)
            self.logger.debug(f"Sending frame {self.frame_counter.value} for processing")
            if not self.async_transform:
                register_transformed_image(self.transform.transform(frame_res))
            else:
                self.transform.transform_async(frame_res, time.time() * 1000 - self.start_time)
        frame_shape_shm.close()
        processed_frame_shm.close()
        frame_shm.close()

    def run(self):
        self.exit_flag = multiprocessing.Value(ctypes.c_bool, False)
        self.shared_mem_lock = multiprocessing.Lock()

        self.resolution_set_event = multiprocessing.Event()
        self.frame_buffer_init_event = multiprocessing.Event()
        self.frame_counter = multiprocessing.Value(ctypes.c_int, 0)


        frame_shape_shm = SharedMemory(name="frame_shape", create=True, size=3*4) #4 bytes per dim as long as int32 is big enough
        frame_shape = np.ndarray(2, buffer=frame_shape_shm.buf, dtype='i4')  #4 bytes per dim as long as int32 is big enough


        transformation_process = multiprocessing.Process(
            target=self.image_process_loop,
            args=()
        )
        webcam_reader_process = multiprocessing.Process(
            target=self.webcam_reader_loop,
            args=()
        )

        webcam_reader_process.start()
        self.resolution_set_event.wait()

        frame_buffer_shm = SharedMemory(name='cur_frame', create=True, size=frame_shape[0] * frame_shape[1] * 3)
        processed_frame_buffer_shm = SharedMemory(name='cur_processed_frame', create=True, size=frame_shape[0] * frame_shape[1] * 3)
        self.frame_buffer_init_event.set()

        transformation_process.start()
        self.opencv_window_loop()
        transformation_process.join()
        webcam_reader_process.join()

        frame_buffer_shm.close()
        frame_shape_shm.close()
        frame_buffer_shm.unlink()
        frame_shape_shm.unlink()
        processed_frame_buffer_shm.close()
        processed_frame_buffer_shm.unlink()