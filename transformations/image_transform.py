import numpy as np

class ImageTransform:
    """
    Interface for transforming images from a stream
    """
    def transform(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("transform() not implemented")