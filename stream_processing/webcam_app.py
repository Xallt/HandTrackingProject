import cv2
import logging
import colorlog
import multiprocessing
import ctypes
import time
from transformations.image_transform import ImageTransform

class WebcamApp:
    def __init__(self, url: str, debug: bool = False):
        self._init_logging(debug)
        self.url = self._preprocess_url(url)
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