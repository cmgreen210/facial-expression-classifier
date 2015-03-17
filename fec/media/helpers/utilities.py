import numpy as np
from PIL import Image
import cv2


def load_np_clf_data(path, fraction=None, seed=None):
    """

    :param path:
    :param fraction:
    :param seed:
    :return:
    """
    files = np.load(path)

    images = files['images']
    targets = files['targets']

    if fraction is not None:
        n = targets.shape[0]
        idx = get_sub_sample_idx(fraction, n, seed)
        images = images[idx]
        targets = targets[idx]

    return images, targets


def get_sub_sample_idx(fraction, n, seed=None):
    """

    :param fraction:
    :param n:
    :param seed:
    :return:
    """
    if fraction > 1 or fraction <= 0:
        raise ValueError('fraction must be in (0, 1]!')

    cnt = np.ceil(n * fraction)

    if seed is not None:
        np.random.seed(seed)

    return np.random.choice(n, cnt, replace=False)


def assemble_dataset(train_path, validation_path, test_path,
                     fraction=None, seed=None):
    """

    :param train_path:
    :param validation_path:
    :param test_path:
    :param fraction:
    :param seed:
    :return:
    """
    dataset = [None, None, None]
    dataset[0] = load_np_clf_data(train_path, fraction=fraction, seed=seed)
    dataset[1] = load_np_clf_data(validation_path,
                                  fraction=fraction, seed=seed)
    dataset[2] = load_np_clf_data(test_path, fraction=fraction, seed=seed)
    return dataset


def display_image(img):
    """

    :param img:
    :return:
    """
    Image.fromarray(np.uint8(img)).show()


def flip_image(image, dir='h'):
    """

    :param image:
    :param dir:
    :return:
    """
    if dir == 'h':
        return image[:, ::-1]
    elif dir == 'v':
        return image[::-1, :]
    else:
        raise ValueError('''Direction must be 'h' or 'v''''')


def get_rotation_matrix(cols, rows, degrees=0, scaling=1):
    """

    :param cols:
    :param rows:
    :param degrees:
    :param scaling:
    :return:
    """
    m = cv2.getRotationMatrix2D((cols/2, rows/2), degrees, scaling)
    return m


def rotate_image(image, rot_mat):
    """

    :param image:
    :param rot_mat:
    :return:
    """
    return cv2.warpAffine(image, rot_mat, image.shape)


if __name__ == '__main__':
    img, targets = load_np_clf_data('npdata/privatetest.npz')
    # m = get_rotation_matrix(48, 48, degrees=45)
    # rot_img = rotate_image(img[0].reshape(48, 48), m)
    # display_image(rot_img)
