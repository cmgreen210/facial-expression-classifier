from fec.media.image_processing import FaceDetectorProcessor
import graphlab as gl
import numpy as np


class ImageFileClassifier(object):
    """Classify stored images

    Parameters
    ----------
    classifier: image classifier
    image_processor: ImageProcessor object that processes any image before
                     classification
    """

    def __init__(self, classifier, image_processor=FaceDetectorProcessor()):
        self._classifier = classifier
        self._image_processor = image_processor

    def classify(self, path, h=48, w=48, channels=1):
        """Classify the image

        :param path: path to image file
        :param h: image height
        :param w: image width
        :param channels: number of channels for the image
        :return: classifications
        """
        image = gl.Image(path)
        data = image.pixel_data.copy()
        image, face = self._image_processor.process_image(data)
        if face is None:
            return None

        face = face.flatten()
        face = face - np.mean(face)
        face /= np.std(face)

        fmin = np.min(face)
        fmax = np.max(face)

        face = np.floor(255 * (face - fmin) / (fmax - fmin))

        face_arr = gl.SArray([face.tolist()])
        clf_image = face_arr.pixel_array_to_image(h, w, channels,
                                                  allow_rounding=True)
        # x = gl.SFrame({'images': clf_image})
        x = np.expend_dims(face, axis=0)
        classifications = self._classifier(x)
        return image, clf_image[0], classifications
