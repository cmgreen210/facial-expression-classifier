import cv2
import time
import os
from fec.classifier.face_detector import FaceDetector
from math import floor


class ImageProcessor(object):
    def __init__(self):
        pass

    def process_image(self, image, *args):
        #  self.save_image(image)
        return image

    def save_image(self, image):
        file_name = os.path.dirname(__file__)
        dir_name = os.path.join(file_name, 'tmp')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        file_name = os.path.join(dir_name,
                                 'test_' + str(int(100000 *
                                                   time.time())) + '.png')
        cv2.imwrite(file_name, image)


class GrayScaleProcessor(ImageProcessor):
    def __init__(self):
        pass

    def process_image(self, image, *args):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray


class ResizeProcessor(ImageProcessor):
    def __init__(self, dim=(48, 48), interp=cv2.INTER_AREA):
        self.dim = dim
        self.interp = interp

    def process_image(self, image, *args):
        dst = cv2.resize(image, self.dim,
                         interpolation=self.interp)
        return dst


class FaceDetectorProcessor(ImageProcessor):

    def __init__(self, cascade_file='haarcascade_frontalface_alt.xml',
                 return_largest=True, scale_x=1.5, scale_y=1.5, shape=None,
                 add_rectangle=True, rect_color=(222, 192, 91)):
        self.detector = FaceDetector(cascade_file,
                                     return_largest=return_largest)

        self.preprocessor = GrayScaleProcessor()
        self.postprocessor = ResizeProcessor()
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.shape = shape

        self.add_rectangle = add_rectangle
        self.rect_color = rect_color

    def process_image(self, image, *args):
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


def run_face_detector(image):
    detector = FaceDetectorProcessor()
    detector.process_image(image)


if __name__ == '__main__':
    fd_proces = FaceDetectorProcessor()
    img = cv2.imread('/Users/chris/face-emotion-classifier/media/ZoyNE2Ww5Z/1-frame.png')
    img, _ = fd_proces.process_image(img)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()