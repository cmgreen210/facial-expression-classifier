import cv
import cv2
from multiprocessing.pool import ThreadPool
from abc import ABCMeta, abstractmethod
from collections import deque
from fec.media.image_processing import FaceDetectorProcessor
import graphlab as gl


class VideoStreamClassifyBase(object):
    """Base class for video expression classification

    Parameters
    ----------
    classifer: the Python underlying classifier
    frame_skip: the number of frames to skip before classification
    image_processor: the image processor to use before sending
                     to the classifier
    """
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
        """Return frame predictions

        :return: array of classification
        """
        return self._classifications

    def _setup_multithreaded(self):
        self.thread_num = cv2.getNumberOfCPUs()
        self.thread_pool = ThreadPool(self.thread_num)

    def process_frame(self, frame, frame_count):
        """Process the input frame

        :param frame: array of pixels
        :param frame_count: the count of the frame
        """
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
        """Return the internal classifier

        :return: classifier object
        """
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        """Set the classifier

        :param classifier: a classifier object
        """
        self._classifier = classifier

    @property
    def frame_skip(self):
        """Get the number of frames to skip between processing

        :return: int the frame count
        """
        return self._frame_skip

    @frame_skip.setter
    def frame_skip(self, frame_skip):
        """Set the number of frames to skip between processing

        :param frame_skip: int number of frames to skip
        """
        if frame_skip <= 0:
            raise ValueError('Frame skip most be positive!')

        self._frame_skip = frame_skip

    @abstractmethod
    def start(self):
        """Start the video stream

        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the video stream

        """
        pass

    @abstractmethod
    def clean_up(self):
        """Clean up the object before stopping

        """
        pass


class CameraClassifier(VideoStreamClassifyBase):
    """Classify images from web camera

    Parameters
    ----------
    classifier: classifier object
    frame_skip: int number of frames to skip between classifications
    source: int camera source
    name: str name of displayed window
    """
    def __init__(self, classifier, frame_skip=20, source=0, name=""):
        super(CameraClassifier, self).__init__(classifier, frame_skip)
        self._source = source
        self._capture = None
        self._name = name

    def start(self):
        """Start the camera and begin classifying

        """
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
        """Stop the camera stream by pressing 'q'

        """
        return cv2.waitKey(1) & 0xFF == ord('q')

    def clean_up(self):
        """Clean up the camera

        """
        if self._capture:
            self._capture.release()

        cv.DestroyAllWindows()

    def _display_image(self, image):
        cv2.imshow(self._name, image)
        return


VideoStreamClassifyBase.register(CameraClassifier)


class VideoFileClassifier(VideoStreamClassifyBase):
    """Video classifier from video file

    Parameters
    ----------
    classifier: classification object
    source: path to video file
    name: str name of displayed window
    frame_skip: the number of frames to skip before classification
    h: height of image
    w: width of image
    d: number of image channels
    """
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
        """Start processing video

        """
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
        """Not used

        """
        pass

    def clean_up(self):
        """Clean up after video image capturing

        This is where the actual classification is done
        """
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

    def get_final_images(self):
        """Return classified images and their processed versions

        :return: tuple (original images, transformed images)
        """
        return self.original_images, self.transformed_images

    @property
    def source(self):
        """Get the source of the video

        :return: path to source
        """
        return self.source

    @source.setter
    def source(self, source):
        """Set the video source path

        :param source:
        """
        self.source = source

VideoStreamClassifyBase.register(VideoFileClassifier)
