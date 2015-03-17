import cv2
import time
import os
from fec.classifier.face_detector import FaceDetector
from math import floor


class ImageProcessor(object):
    """Base class for image processing
    """
    def __init__(self):
        pass

    def process_image(self, image, *args):
        """Returns image

        :param image:
        :param args:
        :return: image the image you put in
        """
        return image

    def save_image(self, image):
        """Saves the image to a temporary directory relative to this file

        :param image: an openv image
        """
        file_name = os.path.dirname(__file__)
        dir_name = os.path.join(file_name, 'tmp')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        file_name = os.path.join(dir_name,
                                 'test_' + str(int(100000 *
                                                   time.time())) + '.png')
        cv2.imwrite(file_name, image)


class GrayScaleProcessor(ImageProcessor):
    """Extends the base processor by converting an image to grayscale
    """
    def __init__(self):
        pass

    def process_image(self, image):
        """Convert the image to grayscale

        :param image: image as an array
        :return: image converted to grayscale
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray


class ResizeProcessor(ImageProcessor):
    """A simple image resizer

    Parameters
    ----------
    :param dim: resize dimensions
    :param interp: cv interpolation enum
    """
    def __init__(self, dim=(48, 48), interp=cv2.INTER_AREA):
        self.dim = dim
        self.interp = interp

    def process_image(self, image):
        """Process the image by resizing

        :param image: the image array
        """
        dst = cv2.resize(image, self.dim,
                         interpolation=self.interp)
        return dst


class FaceDetectorProcessor(ImageProcessor):
    """Image processor that detects attempts to detect a single face

    Parameters
    ----------
    cascade_file: path to haar cascade opencv trained file
    scale_x: amount to scale rectangle of face found in horizontal direction
    scale_y: amount to scale rectangle of face found in vertical direction
    add_rectangle: bool should we add the found face rectangle to the image
    rect_color: the color of the face rectangle outline

    """
    def __init__(self, cascade_file='haarcascade_frontalface_alt.xml',
                 scale_x=1.5, scale_y=1.5,
                 add_rectangle=True, rect_color=(222, 192, 91)):
        self.detector = FaceDetector(cascade_file)

        self.preprocessor = GrayScaleProcessor()
        self.postprocessor = ResizeProcessor()
        self.scale_x = scale_x
        self.scale_y = scale_y

        self.add_rectangle = add_rectangle
        self.rect_color = rect_color

    def process_image(self, image, *args):
        """

        :param image:
        :param args:
        :return:
        """
        gray = self.preprocessor.process_image(image, *args)

        face = self.detector.detect_face(image)
        if len(face) == 0:
            return image, None
        x, y, w, h = face[0]
        w_new = int(floor(w * self.scale_x))
        h_new = int(floor(h * self.scale_y))

        y = int(max(y - floor((h_new - h)/2), 0))
        x = int(max(x - floor((w_new - w)/2), 0))

        gray = gray[y:y+h_new, x:x+w_new]
        gray = self.postprocessor.process_image(gray)

        if self.add_rectangle:
            cv2.rectangle(image, (x, y), (x + w_new, y + h_new),
                          self.rect_color, 2)

        return image, gray
