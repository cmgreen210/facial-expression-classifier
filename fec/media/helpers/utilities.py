import numpy as np
from PIL import Image
import cv2


def load_np_clf_data(path, fraction=None, seed=None):
    """Load compressed numpy model data

    :param path: path to data
    :param fraction: fraction of data to randomly select
    :param seed: random seed
    :return: tuple of images and corresponding target labels
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
    """Get random subsample indices

    :param fraction: fraction of subsample
    :param n: total sample seed
    :param seed: random seed
    :return: array of indices
    """
    if fraction > 1 or fraction <= 0:
        raise ValueError('fraction must be in (0, 1]!')

    cnt = np.ceil(n * fraction)

    if seed is not None:
        np.random.seed(seed)

    return np.random.choice(n, cnt, replace=False)


def assemble_dataset(train_path, validation_path, test_path,
                     fraction=None, seed=None):
    """Create tuple dataset of training, validation and testing data

    :param train_path: path of training data
    :param validation_path: path of validation data
    :param test_path: path of testing data
    :param fraction: amount of each category to sample
    :param seed: random number seed
    :return: tuple of training, validation and testing data
    """
    dataset = [None, None, None]
    dataset[0] = load_np_clf_data(train_path, fraction=fraction, seed=seed)
    dataset[1] = load_np_clf_data(validation_path,
                                  fraction=fraction, seed=seed)
    dataset[2] = load_np_clf_data(test_path, fraction=fraction, seed=seed)
    return dataset


def display_image(img):
    """Display image from numpy array

    :param img: numpy array
    """
    Image.fromarray(np.uint8(img)).show()


def flip_image(image, dir='h'):
    """Flip image (numpy array)

    :param image: array
    :param dir: direction to flip, either horizontal 'h' or vertical 'v'
    :return:
    """
    if dir == 'h':
        return image[:, ::-1]
    elif dir == 'v':
        return image[::-1, :]
    else:
        raise ValueError('''Direction must be 'h' or 'v''''')


def get_rotation_matrix(cols, rows, degrees=0, scaling=1):
    """Get an image rotation matrix

    :param cols: target cols
    :param rows: target rows
    :param degrees: degrees of rotation
    :param scaling: amount of scaling
    :return: rotation matrix
    """
    m = cv2.getRotationMatrix2D((cols/2, rows/2), degrees, scaling)
    return m


def rotate_image(image, rot_mat):
    """Rotate image

    :param image: array image
    :param rot_mat: rotation array
    :return: rotated image
    """
    return cv2.warpAffine(image, rot_mat, image.shape)


if __name__ == '__main__':
    img, targets = load_np_clf_data('npdata/privatetest.npz')
    # m = get_rotation_matrix(48, 48, degrees=45)
    # rot_img = rotate_image(img[0].reshape(48, 48), m)
    # display_image(rot_img)
