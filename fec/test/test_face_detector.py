import unittest
import nose.tools as no
from fec.classifier.face_detector import *
import cv2
import os


def _image_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class TestFaceDetector(unittest.TestCase):

    def setUp(self):
        self.dir = os.path.dirname(__file__)

    @no.raises(IOError)
    def test_init_fail(self):
        """Test if initialization fails if config file not found"""
        _ = FaceDetector('no_file!')

    def test_init_ok(self):
        """Test if simple construction of detector works"""
        _ = FaceDetector('haarcascade_frontalface_alt.xml')

    def test_detect_faces(self):
        """Test if we can detect two faces"""
        fd = FaceDetector('haarcascade_frontalface_default.xml')

        path = os.path.join(self.dir, 'data', 'sg.jpg')
        img = cv2.imread(path)
        img = _image_to_gray(img)
        faces = fd.detect_face(img)
        n = len(faces)
        no.assert_equal(n, 2, 'Expected to find 2 faces! Got %d!' % n)

        fd.return_largest = True
        face = fd.detect_face(img)
        n = len(face)
        no.assert_equal(n, 1, 'Expected only the biggest face!')

        path = os.path.join(self.dir, 'data', 'parakeet.jpg')
        img_no_face = cv2.imread(path)
        gray_no = _image_to_gray(img_no_face)
        fd.min_size = (60, 60)
        faces = fd.detect_face(gray_no)
        no.assert_equal(len(faces), 0, 'Expected no faces!')
