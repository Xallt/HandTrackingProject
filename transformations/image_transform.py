import numpy as np

class ImageTransform:
    """
    Interface for transforming images from a stream
    """
    def transform(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("transform() not implemented")
    def set_callback(self, callback):
        self.callback = callback
    def transform_async(self, image: np.ndarray, timestamp: float):
        """
        Asynchronous transform of image
        Default implementation just calls the transform() method
        """
        self.callback(self.transform(image))