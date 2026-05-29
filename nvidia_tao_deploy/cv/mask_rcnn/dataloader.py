# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MRCNN loader."""

import numpy as np
from PIL import Image

from nvidia_tao_deploy.dataloader.coco import COCOLoader
from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input


class MRCNNCOCOLoader(COCOLoader):
    """MRCNN DataLoader."""

    def preprocess_image(self, image_path):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * MRCNN: Resizes and pads the image to fit the input size.

        Args:
            image_path(str): The path to the image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
            scale (list): the resize scale used, if any.
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """Resize and Pad.

            A subroutine to implement padding and resizing. This will resize the image to fit
            fully within the input size, and pads the remaining bottom-right portions with
            the value provided.

            Args:
                image (PIL.Image): The PIL image object
                pad_color (list): The RGB values to use for the padded area. Default: Black/Zeros.

            Returns:
                pad (PIL.Image): The PIL image object already padded and cropped,
                scale (list): the resize scale used.
            """
            width, height = image.size
            width_scale = width / self.width
            height_scale = height / self.height
            scale = 1.0 / max(width_scale, height_scale)
            image = image.resize(
                (round(width * scale), round(height * scale)),
                resample=Image.BILINEAR)
            pad = Image.new("RGB", (self.width, self.height))
            pad.paste(pad_color, [0, 0, self.width, self.height])
            pad.paste(image)
            return pad, scale

        scale = None
        image = Image.open(image_path)
        image = image.convert(mode='RGB')

        # zero pad
        image, scale = resize_pad(image, (124, 116, 104))
        image = np.asarray(image, dtype=self.dtype)

        if self.data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        # Normalize and apply imag mean and std
        image = preprocess_input(image, data_format=self.data_format, mode='torch')

        return image, scale
