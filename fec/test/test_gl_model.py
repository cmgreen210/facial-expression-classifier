import unittest
from fec.classifier.gl_classifier import GraphLabClassifierFromFile
import os
import graphlab as gl


class GraphlabClassifierTest(unittest.TestCase):
    def test_graphlab_classifier(self):
        """ Test the graphlab clasifier from file

        :return:
        """

        this_dir, _ = os.path.split(os.path.abspath(__file__))
        this_dir = os.path.abspath(this_dir)
        model_path = os.path.join(this_dir, 'data', 'gl_mdl')

        model = GraphLabClassifierFromFile(model_path)
        self.assertEqual(model._model.name(),
                         'NeuralNetClassifier')

        x = gl.load_sframe(os.path.join(this_dir,
                                        'data',
                                        'img_10'))
        pred = model.predict(x)
        self.assertEqual(len(pred), 10)
        pred_prob = model.predict_proba(x)
        rows, cols = pred_prob.shape
        self.assertEqual(rows, 10 * 7)
        self.assertEqual(cols, 3)
