from classifier_base import ClassifierBase
import os
import graphlab as gl
import numpy as np
from math import floor


class GraphLabClassifierFromFile(ClassifierBase):
    """

    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise ValueError("Model path does not exist!")

        self._model = gl.load_model(model_path)
        self._num_class = self._model['num_classes']

    def fit(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        pass

    def predict(self, x):
        """

        :param x:
        :return:
        """
        return self._model.predict(x)

    def predict_proba(self, x):
        """

        :param x:
        :return:
        """
        return self._model.predict_topk(x, k=self._num_class)


class GraphLabClassifierFromNetBuilder(ClassifierBase):
    """

    """
    def __init__(self, net_builder, train_frac=.8, h=48, w=48, depth=1,
                 target='label', feat_name='images',
                 max_iterations=50, verbose=True, chkpt_dir=''):
        self._net_builder = net_builder
        self._model = None

        self._feat_means = None
        self._feat_std = None

        self._h = h
        self._w = w
        self._d = depth

        self._target = target
        self._feat_name = feat_name

        # Fit parameters
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._chkpt_dir = chkpt_dir
        self._train_frac = train_frac

    def _create_images(self, x):
        sarray = gl.SArray(x)
        images = sarray.pixel_array_to_image(self._w, self._h, self._d,
                                             allow_rounding=True)
        return images

    def _scale_features(self, x):

        row_means = np.mean(x, axis=1)
        x -= row_means[:, np.newaxis]

        x -= self._feat_means
        x /= self._feat_std

        # Scale back to [0, 255]
        x_min = np.min(x)
        x_max = np.max(x)

        return 255 * (x - x_min) / (x_max - x_min)

    def _assemble_full_dataset(self, x, y):
        images = self._create_images(x)
        sf = gl.SFrame({self._feat_name: images,
                        self._target: y})
        return sf

    def _split(self, x, y):
        n_examples = x.shape[0]
        idx = np.random.permutation(n_examples)
        n_test = floor(self._train_frac * n_examples)

        return (x[idx[:n_test], :], y[idx[:n_test]],
                x[idx[n_test:], :], y[idx[n_test:]])

    def fit(self, x, y, **kwargs):
        """

        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        x_train, y_train, x_valid, y_valid = self._split(x, y)
        self._feat_means = np.mean(x_train, axis=0)
        self._feat_std = np.std(x_train, axis=0)

        x_train = self._scale_features(x_train)
        x_valid = self._scale_features(x_valid)

        train_set = self._assemble_full_dataset(x_train, y_train)
        valid_set = self._assemble_full_dataset(x_valid, y_valid)

        # Time to train the model
        self._model = gl.neuralnet_classifier.create(
            train_set,
            network=self._net_builder.get_net(),
            target=self._target,
            max_iterations=self._max_iterations,
            model_checkpoint_path=self._chkpt_dir,
            verbose=self._verbose,
            validation_set=valid_set,
            metric=['accuracy',
                    'error',
                    'recall@1'
                    ]
        )

        return

    def _create_gl_feature_mat(self, x):
        scale_x = self._scale_features(x)
        images = self._create_images(scale_x)
        sf = gl.SFrame({self._feat_name: images})
        return sf

    def predict(self, x):
        """

        :param x:
        :return:
        """
        return self._model.predict(
            self._create_gl_feature_mat(x))

    def predict_proba(self, x, k=3):
        """

        :param x:
        :param k:
        :return:
        """
        return self._model.predict_topk(
            self._create_gl_feature_mat(x), k=k)

    def evaluate(self, x, y, metric='auto'):
        """

        :param x:
        :param y:
        :param metric:
        :return:
        """
        scale_x = self._scale_features(x)
        dataset = self._assemble_full_dataset(scale_x, y)
        return self._model.evaluate(dataset, metric=metric)
