import cv
import cv2
from multiprocessing.pool import ThreadPool
import image_processing as imp
from collections import deque
from abc import ABCMeta, abstractmethod
from collections import deque
from fec.media.image_processing import FaceDetectorProcessor
from fec.classifier.gl_classifier import GraphLabClassifierFromFile
import os
import graphlab as gl
import shutil
import numpy as np


class VideoStreamClassifyBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, classifier, frame_skip=20,
                 image_processor=FaceDetectorProcessor()):
        self._classifier = classifier
        self._frame_skip = None
        self.frame_skip = frame_skip
        self._classifications = []
        self.images = []
        self.thread_num = None
        self.thread_pool = None
        self._setup_multithreaded()
        self.tasks = deque()
        self.image_processor = image_processor

        self.original_images = None
        self.transformed_images = None
        self.image_paths = None

    def get_classifications(self):
        return self._classifications

    def _setup_multithreaded(self):
        self.thread_num = cv2.getNumberOfCPUs()
        self.thread_pool = ThreadPool(self.thread_num)

    def process_frame(self, frame, frame_count):
        while len(self.tasks) > 0 and self.tasks[0].ready():
            self.images.append(self.tasks.popleft().get())

        if len(self.tasks) < self.thread_num and\
           frame_count % self.frame_skip == 0:
            task_func = self.image_processor.process_image
            task = self.thread_pool.apply_async(task_func,
                                                (frame.copy(),))
            self.tasks.append(task)

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        self._classifier = classifier

    @property
    def frame_skip(self):
        return self._frame_skip

    @frame_skip.setter
    def frame_skip(self, frame_skip):
        if frame_skip <= 0:
            raise ValueError('Frame skip most be positive!')

        self._frame_skip = frame_skip

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def clean_up(self):
        pass


class CameraClassifier(VideoStreamClassifyBase):
    def __init__(self, classifier, frame_skip=20, source=0, name=""):
        super(CameraClassifier, self).__init__(classifier, frame_skip)
        self._source = source
        self._capture = None
        self._name = name

    def start(self):

        self._capture = cv2.VideoCapture(self._source)
        frame_count = 0
        while self._capture.isOpened():
            ret, frame = self._capture.read()
            frame_count += 1

            if self.stop():
                break

            self._display_image(frame)

            self.process_frame(frame, frame_count)

        self.clean_up()

    def stop(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def clean_up(self):
        if self._capture:
            self._capture.release()

        cv.DestroyAllWindows()

    def get_classifications(self):
        pass

    def _display_image(self, image):
        cv2.imshow(self._name, image)
        return


VideoStreamClassifyBase.register(CameraClassifier)


class VideoFileClassifier(VideoStreamClassifyBase):
    def __init__(self, classifier, source, frame_skip=20, name="",
                 h=48, w=48, d=1):
        super(VideoFileClassifier, self).__init__(classifier, frame_skip)
        self._source = source
        self._capture = None
        self._name = name

        self._h = h
        self._w = w
        self._d = d

    def start(self):

        self._capture = cv2.VideoCapture(self._source)
        frame_count = 0
        while self._capture.isOpened():
            ret, frame = self._capture.read()
            frame_count += 1

            if not ret:
                break
            self.process_frame(frame, frame_count)

        self.clean_up()

    def stop(self):
        pass

    def clean_up(self):
        if self._capture:
            self._capture.release()

        self.transformed_images = None
        self.original_images = None

        if self.images is not None:
            count = 0

            self.transformed_images = []
            self.original_images = []

            images_to_gl = []
            for im in self.images:
                if im[0] is None or im[1] is None:
                    continue

                self.original_images.append(im[0])
                self.transformed_images.append(im[1])
                images_to_gl.append(im[1].flatten().tolist())
                count += 1

            x = gl.SArray(images_to_gl)
            x.pixel_array_to_image(self._w, self._h, self._d)
            x = gl.SFrame({'images': x})

            if self.classifier is not None:
                self._classifications = self.classifier(x)

    def get_classifications(self):
        return self._classifications

    def get_final_images(self):
        return self.original_images, self.transformed_images

    @property
    def source(self):
        return self.source

    @source.setter
    def source(self, source):
        self.source = source

VideoStreamClassifyBase.register(VideoFileClassifier)


if __name__ == '__main__':
    clf = GraphLabClassifierFromFile('/Users/chris/face-django_expression-classifier'
                                     '/django_expression/classifier')
    vid = VideoFileClassifier(clf.predict_proba,
                              '/Users/chris/face-django_expression-classifier'
                              '/tmp_video/vid.mov')
    vid.start()
    vid.stop()

    print len(vid.original_images)
    print len(vid.transformed_images)
