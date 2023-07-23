import numpy as np

class ImageTransform:
    """
    Interface for transforming images from a stream
    """
    def validate(self, image: np.ndarray) -> bool:
        raise NotImplementedError("validate() not implemented")
    def transform(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("transform() not implemented")