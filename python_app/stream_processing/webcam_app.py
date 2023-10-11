import cv2
import logging
import colorlog
import multiprocessing
import ctypes
import time
import numpy as np

class WebcamApp:
    def __init__(self, url: str, debug: bool = False):
        self._init_logging(debug)
        self.url = self._preprocess_url(url)
        self._stats_memory = {}
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

    def add_stats(
            self, 
            frame: np.ndarray, 
            stats: dict,
            color: tuple=(0, 255, 0),
            precision: int = 4,
            scale=1.0,
            alpha=1.0 # EMA alpha
        ):
        # First, compute the size of the background rectangle
        height, width = 0, 0
        for key, value in stats.items():
            if key in self._stats_memory:
                self._stats_memory[key] = (1 - alpha) * self._stats_memory[key] + alpha * value
                value = self._stats_memory[key]
            else:
                self._stats_memory[key] = value

            if type(value) is int:
                value = str(value)
            elif type(value) is float:
                value = f"{value:.{precision}f}"
            text_size = cv2.getTextSize(
                f"{key}: {value}", 
                cv2.FONT_HERSHEY_SIMPLEX, 
                scale, 
                2
            )[0]
            height += text_size[1] + 10
            width = max(width, text_size[0] + 20)

        height += int(10 * scale)

        # Draw the background rectangle
        cv2.rectangle(
            frame,
            (0, 0),
            (width, height),
            (0, 0, 0),
            cv2.FILLED
        )

        for i, (key, value) in enumerate(stats.items()):
            value = self._stats_memory[key]
            if type(value) is int:
                value = str(value)
            elif type(value) is float:
                value = f"{value:.{precision}f}"
            cv2.putText(
                frame, 
                f"{key}: {value}", 
                (10, int((30 + 40 * i) * scale)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                scale, 
                color,
                2,
                cv2.FILLED  
            )
