from fec.media.image_processing import FaceDetectorProcessor
import graphlab as gl
import numpy as np


class ImageFileClassifier(object):

    def __init__(self, classifier, image_processor=FaceDetectorProcessor()):
        self._classifier = classifier
        self._image_processor = image_processor

    def classify(self, path, h=48, w=48, channels=1):
        image = gl.Image(path)
        data = image.pixel_data.copy()
        image, face = self._image_processor.process_image(data)
        if face is None:
            return None

        face = face.flatten()
        face = face - np
        face /= np.std(face)

        face_arr = gl.SArray([face.tolist()])
        clf_image = face_arr.pixel_array_to_image(h, w, channels)
        x = gl.SFrame({'images': clf_image})
        classifications = self._classifier(x)
        return image, clf_image[0], classifications
