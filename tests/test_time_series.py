import pytest
import numpy as np
import esam


def test_offset_series():
    ser = esam.OffsetSeries(np.array([1,2,3,4]), offset=-2)
    assert np.all(ser.data == np.array([[0, 0, 1, 2]]))

    ser = esam.OffsetSeries(np.array([1,2,3,4]), offset=2)
    assert np.all(ser.data == np.array([[3, 4, 0, 0]]))

def test_offset_series():
    ser = esam.OffsetSeries(np.array([1,2,3,4]), offset=2)
    assert np.all(ser == np.array([[3, 4, 0, 0]]))


    assert ser[0] == 3
    assert ser[1] == 4
    assert ser[2] == 0
