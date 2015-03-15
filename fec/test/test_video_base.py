import unittest
from fec.media.video import VideoStreamClassifyBase
from fec.classifier.classifier_base import DummyClassifier


class DummyVideoStream(VideoStreamClassifyBase):

    def __init__(self, clf, frame_skip=5):
        super(DummyVideoStream, self).__init__(clf, frame_skip=frame_skip)

    def start(self):
        pass

    def stop(self):
        pass

    def clean_up(self):
        pass

    def process_frame(self, frame, frame_count):
        pass


VideoStreamClassifyBase.register(DummyVideoStream)


class VideoBaseTest(unittest.TestCase):

    def test_getters_and_setters(self):
        v_1 = DummyVideoStream(None, frame_skip=5)

        self.assertEqual(v_1.frame_skip, 5)
        with self.assertRaises(ValueError):
            v_1.frame_skip = -1

        frame_skip = 10
        v_1.frame_skip = frame_skip
        self.assertEqual(v_1.frame_skip, frame_skip)

        v_1.classifier = DummyClassifier(0)
        self.assertIsNotNone(v_1.classifier)

    def test_construction(self):
        self.assertRaises(ValueError, DummyVideoStream, None, -1)
        _ = DummyVideoStream(None, 20)
