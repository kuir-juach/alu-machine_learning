#!/usr/bin/env python3
"""
This module for a function to perform a valid convolution on grayscale images.
A batch of grayscale images and 
a kernel, and it returns the convolved output for each image.
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
    images (numpy.ndarray): Shape (m, h, w) containing multiple grayscale images.
        - m is the number of images
        - h is the height in pixels of the images
        - w is the width in pixels of the images
    kernel (numpy.ndarray): Shape (kh, kw) containing the kernel for the convolution.
        - kh is the height of the kernel
        - kw is the width of the kernel

    Returns:
    numpy.ndarray: Containing the convolved images.
    """

    # Get the dimensions of images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the dimensions of the output after valid convolution
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize the output array with zeros
    output = np.zeros((m, output_h, output_w))

    # Perform convolution on each image
    for i in range(m):  # Loop over each image
        for x in range(output_h):  # Loop over the output height
            for y in range(output_w):  # Loop over the output width
                # Extract the current region of the image and apply the kernel
                region = images[i, x:x + kh, y:y + kw]
                output[i, x, y] = np.sum(region * kernel)

    return output

