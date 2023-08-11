from transformations.image_transform import ImageTransform

class NegativeTransformation(ImageTransform):
    def transform(self, image):
        return 255 - image