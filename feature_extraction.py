"""
This module provides methods to extract features from images of symbols and
associate labels categorizing that symbol with the extracted features.
"""
import os
from dataclasses import dataclass
from typing import Generator, List, Tuple

import numpy as np
from PIL.Image import BILINEAR, Image
from pdf2image import convert_from_path

import imhelp

DEFAULT_FEATURE_VECTOR_SIZE_ROOT = 8
''' The square root of the default feature vector length.'''

DEFAULT_WHITE_THRESHOLD = 200
'''The default threshold above which a pixel is considered white.'''


@dataclass(init=True, frozen=True, eq=True, order=False, unsafe_hash=False)
class TrainingGridSpec:
    """Represents a training grid specification. The specification contains
    the value for the shape of a single cell of the grid (width x height) in
    inches, the top left corner of the grid relative to paper `(x,y)` in inches,
    the number of columns in the grid, and the number of rows in the grid."""
    shape: Tuple[int, int]
    top_left: Tuple[int, int]
    num_columns: int
    num_rows: int


DEFAULT_GRID_SPEC = TrainingGridSpec(num_rows=8, num_columns=8,
                                     shape=(1, 1), top_left=(0, 0))
'''The default training grid spec. 8 rows, 8 columns, in a grid of size
1 in. x 1 in. starting at top left corner of the paper.'''


@dataclass(init=True, frozen=True, eq=True, order=False, unsafe_hash=False)
class TrainingFileSpec:
    """
    Represents a training file specification. A training file is expected to be
    in PDF and have pages containing a black border grid with white background
    with each cell of the grid containing an image of a handwritten character.
    The DPI parameter is used to convert inches coordinates in the grid
    specification to pixels. The white threshold is a pixel value above
    which a pixel is classified white.
    """
    samples_file_path: str
    grid_spec: TrainingGridSpec
    label_file_path: str
    dpi: int
    white_threshold: int


class FeatureExtractor:

    @staticmethod
    def get_feature_matrix(feature_vectors: List[np.ndarray]) -> np.ndarray:
        """Given a list of feature vectors returns a matrix
        with each feature vector on one row with the row ordering
        matching the original ordering of the vector in the list.

        :param feature_vectors: The feature vectors.
        :return: The matrix of feature vectors.
        """
        if not feature_vectors:
            return np.zeros((0, 0), dtype=int)

        first_vector = feature_vectors[0]
        matrix_shape = (0, first_vector.shape[1])
        features_matrix = np.zeros(matrix_shape, dtype=first_vector.dtype)
        for lfv in feature_vectors:
            if lfv.shape[1] != first_vector.shape[1]:
                raise IndexError("Given feature vectors must be"
                                 " of the same shape")
            features_matrix = np.vstack([features_matrix, lfv])

        return features_matrix

    @staticmethod
    def extract_features_from_image(
            image: Image,
            img_vec_len=DEFAULT_FEATURE_VECTOR_SIZE_ROOT,
            threshold=DEFAULT_WHITE_THRESHOLD) -> np.ndarray:
        """
        Extracts a features vector from an image. The image is trimmed to remove
        the whitespace and resized to fit in a square of size
        :param:`img_vec_len`.

        :param image:       The image from which features will be extracted.
        :param img_vec_len: The square root of the length of feature vector that
                            is required.
        :param threshold:   The threshold above which a pixel is to be treated
                            as white.
        :return:            The extracted feature vector.
        """
        resized_im = imhelp.remove_border_and_padding(image.convert('L'),
                                                      white_threshold=threshold)
        resized_im = imhelp.make_square(resized_im, side=img_vec_len)
        resize_shape = (1, img_vec_len ** 2)
        resized_im_arr = np.asarray(resized_im).reshape(resize_shape).copy()
        resized_im_arr[resized_im_arr > threshold] = imhelp.WHITE_PIXEL_VAL
        return resized_im_arr

    @staticmethod
    def grid_split(im: Image, dpi: int, white_threshold: int,
                   spec: TrainingGridSpec) -> Generator[Image, None, None]:
        """
        Splits an image of the training grid into a series of images.

        :param im: An image of a training grid.
        :param dpi: The DPI at which extraction should be done.
        :param spec: The grid specification.
        :param white_threshold: The threshold above which a pixel is considered
        white.
        :return: A generator of images extracted from the grid.
        """
        chopsize_x = spec.shape[0] * dpi
        chopsize_y = spec.shape[1] * dpi

        for col in range(0, spec.num_columns):
            for row in range(0, spec.num_rows):
                x0 = (col * dpi) + spec.top_left[0]
                y0 = (row * dpi) + spec.top_left[1]

                box = (x0, y0, x0 + chopsize_x, y0 + chopsize_y)
                if x0 + chopsize_x < im.size[0] and \
                        y0 + chopsize_y < im.size[1]:
                    yield imhelp.remove_border_and_padding(im.crop(box),
                                                           white_threshold)
                else:
                    yield None

    @staticmethod
    def extract_labelled_feature_vectors(
            input_file: TrainingFileSpec,
            rotations: np.ndarray,
            sq_root_feature_vec_size=DEFAULT_FEATURE_VECTOR_SIZE_ROOT) -> \
            Tuple[np.ndarray, List[str]]:
        """
        Extracts features and labels from a given training file.

        :param input_file: The specification of the input training file.
        :param rotations: An array of rotations in degrees which should be
        applied to every symbol.
        :param sq_root_feature_vec_size: The square root of the size of feature
        vector needed.
        :returns: A tuple with first element as a feature matrix of shape (n,m)
        with n feature vectors of length sq_root_feature_vec_size ** 2
        """
        if not os.path.isfile(input_file.samples_file_path):
            raise IOError("File %s not found." % input_file.samples_file_path)
        if not os.path.isfile(input_file.label_file_path):
            raise IOError("File %s not found." % input_file.label_file_path)

        images = convert_from_path(input_file.samples_file_path,
                                   dpi=input_file.dpi,
                                   transparent=False, grayscale=True)
        with open("data/training_labels.txt") as file:
            all_labels = [line.strip() for line in file]

        if len(images) != len(all_labels):
            raise Exception("Number of pages in the training file must match "
                            "the number of labels provided")

        features = []
        labels = []
        for im_idx, extracted_img in enumerate(images):
            for grid_img in FeatureExtractor.grid_split(
                    extracted_img, dpi=input_file.dpi,
                    spec=input_file.grid_spec,
                    white_threshold=input_file.white_threshold):
                label = all_labels[im_idx]
                for angle in rotations:
                    if abs(angle) > 0:
                        grid_img = grid_img.rotate(angle, resample=BILINEAR)

                    lum_arr = FeatureExtractor.extract_features_from_image(
                            grid_img, sq_root_feature_vec_size,
                            threshold=input_file.white_threshold)
                    features.append(lum_arr)
                    labels.append(label)

        return FeatureExtractor.get_feature_matrix(features), labels
