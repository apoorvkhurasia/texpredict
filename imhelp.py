from typing import Tuple

import numpy
from PIL import Image

WHITE_PIXEL_VAL = 255
'''The value of a white pixel.'''


def make_square(img: Image, side: int) -> Image.Image:
    """
    Resizes an image to fit into a square of given side without any distortion
    of aspect ratio.
    :param img:     The image to resize.
    :param side:    The side of the square.
    :return:        The resized image.
    """
    im = numpy.asarray(img)
    shape = im.shape
    excess_width = shape[0] - shape[1]
    if excess_width > 0:
        pad_width = [(0, 0), (0, shape[0] - shape[1])]
        im = numpy.lib.pad(im, pad_width, 'constant',
                           constant_values=255)
    else:
        pad_width = [(0, shape[1] - shape[0]), (0, 0)]
        im = numpy.lib.pad(im, pad_width, 'constant',
                           constant_values=255)

    return Image.fromarray(im).resize(size=(side, side),
                                      resample=Image.BILINEAR)


def remove_border_and_padding(im: Image, rem_black=True,
                              rem_white=True,
                              white_threshold=WHITE_PIXEL_VAL,
                              black_threshold=0) -> Image:
    """
    Removes borders and padding from a given image. Borders are assumed black
    and padding white.
    :param im:          The image to trim.
    :param rem_black:   Set to true if black padding/border is to be removed.
    :param rem_white:   Set to true if white padding/border is to be removed.
    :param black_threshold: The value below which a pixel is classified black
    :param white_threshold: The value above which a pixel is classified white
    :return:            The trimmed image.
    """
    if not rem_black and not rem_white:
        # Save yourself the trouble
        return im

    orig_image_data = numpy.asarray(im)

    # Remove all rows and columns that only have black pixels
    if rem_black:
        nb_cols = numpy.where(orig_image_data.max(axis=0) > black_threshold)[0]
        nb_rows = numpy.where(orig_image_data.max(axis=1) > black_threshold)[0]
        nb_crop_box = (
                min(nb_rows), max(nb_rows), min(nb_cols), max(nb_cols))
        orig_image_data = orig_image_data[nb_crop_box[0]:nb_crop_box[1] + 1,
                                          nb_crop_box[2]:nb_crop_box[3] + 1]

    # Remove all rows and columns that only have white pixels
    if rem_white:
        nw_cols = numpy.where(orig_image_data.min(axis=0) < white_threshold)[0]
        nw_rows = numpy.where(orig_image_data.min(axis=1) < white_threshold)[0]
        left, right = (0, orig_image_data.shape[1] - 1)
        top, bottom = (0, orig_image_data.shape[0] - 1)
        if nw_cols.any():
            left, right = (min(nw_cols), max(nw_cols))
        if nw_rows.any():
            top, bottom = (min(nw_rows), max(nw_rows))

        nw_crop_box = (top, bottom, left, right)
        orig_image_data = orig_image_data[nw_crop_box[0]:nw_crop_box[1] + 1,
                                          nw_crop_box[2]:nw_crop_box[3] + 1]

    return Image.fromarray(orig_image_data)


def extract_leftmost_symbol(im: Image, white_threshold=WHITE_PIXEL_VAL) ->\
        Tuple[Image.Image, Image.Image]:
    """
    Splits an image into left and right so that the left portion contains
    one or zero symbols. Images are split by vertical whitespace.
    :param im:      The image to split.
    :param white_threshold: The value above which a pixel is classified white
    :return:        A tuple with first element as the left image and the second
                    element as the right image.
    """
    img_data = numpy.asarray(im)
    all_while_cols = numpy.where(img_data.min(axis=0) >= white_threshold)[0]
    if all_while_cols.any():
        next_white_col = min(all_while_cols)
        lt_crop_box = (0, img_data.shape[0] - 1, 0, next_white_col)
        lt_img = Image.fromarray(
                img_data[lt_crop_box[0]:lt_crop_box[1] + 1,
                         lt_crop_box[2]:lt_crop_box[3] + 1])
        if next_white_col < img_data.shape[1]:
            rt_crop_box = (0, img_data.shape[0] - 1,
                           next_white_col, img_data.shape[1] - 1)
            rt_img = Image.fromarray(
                        img_data[rt_crop_box[0]:rt_crop_box[1] + 1,
                                 rt_crop_box[2]:rt_crop_box[3] + 1])
        else:
            rt_img = None
    else:
        lt_img = Image.fromarray(img_data)
        rt_img = None

    return lt_img, rt_img
