from fec.media.helpers.utilities import *
import os
import nose.tools as no
import numpy.testing as nptest
import unittest


class TestUtilities(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.dirname(__file__)

    def test_load_np_clf_data(self):
        path = os.path.join(self.dir,
                            'data',
                            'privatetest.npz')
        images, target = load_np_clf_data(path)
        no.assert_equal(images.shape[0], target.shape[0])

    def test_get_sub_sample_idx(self):
        n = 10
        frac = .2
        no.assert_equal(len(get_sub_sample_idx(frac, n)),
                        2)

    @no.raises(ValueError)
    def test_get_sub_sample_idx_BAD_1(self):
        get_sub_sample_idx(-.01, 10)

    @no.raises(ValueError)
    def test_get_sub_sample_idx_BAD_2(self):
        get_sub_sample_idx(1.1, 10)

    @no.raises(ValueError)
    def test_flip_image_bad_input(self):
        flip_image([], 'bad')

    def test_flip_image(self):
        img = np.array([[1, 2], [3, 4]])
        img_flip_h = np.array([[2, 1], [4, 3]])
        nptest.assert_equal(flip_image(img, 'h'), img_flip_h)

        img_flip_v = np.array([[3, 4], [1, 2]])
        nptest.assert_equal(flip_image(img, 'v'), img_flip_v)
