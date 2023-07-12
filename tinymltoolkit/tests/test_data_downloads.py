import pytest
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tinymltoolkit import data_download


def test_create_color_map():
    hex_list = ['#FF0000', '#00FF00', '#0000FF']

    color_map = data_download.create_color_map(hex_list)

    # Check if the returned object is an instance of LinearSegmentedColormap
    assert isinstance(color_map, LinearSegmentedColormap)

    # Check if the number of colors in the color map matches the input list
    assert len(color_map.colors) == len(hex_list)

    # Check if the color map is created correctly by comparing specific colors
    assert np.allclose(color_map(0.0), (1.0, 0.0, 0.0, 1.0))  # First color should be red
    assert np.allclose(color_map(0.5), (0.0, 1.0, 0.0, 1.0))  # Middle color should be green
    assert np.allclose(color_map(1.0), (0.0, 0.0, 1.0, 1.0))  # Last color should be blue


def test_calculate_square_corners():
    pointlat = 10.0
    pointlong = 20.0
    pad = 0.1

    toplat, toplong, botlat, botlong = data_download.calculate_square_corners(pointlat, pointlong, pad)

    # Check if the calculated values are correct
    assert toplat == 10.1
    assert toplong == 20.1
    assert botlat == 9.9
    assert botlong == 19.9

    # Check if changing the padding affects the calculated values
    new_pad = 0.05
    toplat, toplong, botlat, botlong = calculate_square_corners(pointlat, pointlong, new_pad)
    assert toplat == 10.05
    assert toplong == 20.05
    assert botlat == 9.95
    assert botlong == 19.95
