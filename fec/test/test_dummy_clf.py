import unittest
from fec.classifier.classifier_base import DummyClassifier
import numpy as np


class DummyClassifierTest(unittest.TestCase):
    def test_dummy_classifier(self):
        """Test dummy classifier

        :return:
        """
        n = 10
        dummy = DummyClassifier(n)

        size = (100, 4)
        x = np.random.random(size)
        dummy.fit(x, None)
        predict = dummy.predict(x)
        self.assertEqual(predict.shape[0], size[0])
        self.assertTrue(np.all(predict < n))
        self.assertTrue(np.all(predict >= 0))

        prob = dummy.predict_proba(x)
        self.assertEqual(prob.shape[0], x.shape[0])
        self.assertEqual(prob.shape[1], n)
        self.assertEqual(prob[0, 0], 0.1)
