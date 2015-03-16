import cv2
import os


class FaceDetector(object):
    """Face detection class using opencv CascadeClassifier

    The FaceDetector class wraps the opencv CascadeClassifier. The user
    must provide a path to an xml file with the cascade parameters.

    Parameters
    ----------
    cascade_file : path to the xml cascade classifier parameters
    scale_factor : float specifying reduction in image at each scale
    min_neighbors : minimum number of neighbors needed to retain a rectangle
    min_size : minimum possible object size
    return_largest : returns the largest found face
    """
    def __init__(self, cascade_file, scale_factor=1.1,
                 min_neighbors=5, min_size=(20, 20),
                 return_largest=False):

        self.cascade_file = cascade_file
        self.cascade_classifier = None
        self._set_up_face_detector()

        # Detection parameters
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.return_largest = return_largest

    def _set_up_face_detector(self):
        """Set up the opencv haar face detector"""
        path = os.path.dirname(__file__)
        path = os.path.join(path, 'data', self.cascade_file)
        if not os.path.exists(path):
            err_str = '''Haar cascade file %s does\
                        not exist!''' % self.cascade_file
            err_str.format(self.cascade_file)
            raise IOError(err_str)

        self.cascade_classifier = cv2.CascadeClassifier(path)

    def detect_face(self, image):
        """Detect faces in the image

        :param image: image to detect face in (numpy area)
        :return: list of tuples of (x, y, w, h) for each face found
        """
        clf = self.cascade_classifier
        faces = clf.detectMultiScale(image,
                                     scaleFactor=self.scale_factor,
                                     minNeighbors=self.min_neighbors,
                                     minSize=self.min_size,
                                     flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        if len(faces) == 0:
            return []

        if not self.return_largest:
            return faces

        iter_faces = iter(faces)
        max_face = iter_faces.next()
        max_area = max_face[2] * max_face[3]
        for next_face in iter_faces:
            next_area = next_face[2] * next_face[3]
            if next_area > max_area:
                max_area = next_area
                max_face = next_face

        return [max_face]
