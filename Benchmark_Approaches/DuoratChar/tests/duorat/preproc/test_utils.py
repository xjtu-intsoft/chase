import pytest
import torch
import random

from duorat.preproc.utils import pad_nd_tensor


@pytest.fixture(params=[tuple(), (1,), (0,), (307,), (307, 1)])
def trailing_dims(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 47, 257])
def shapes(request):
    return [
        (random.randint(0, 23), random.randint(0, 31)) for _ in range(request.param)
    ]


def test_pad_2d_sequence(shapes, trailing_dims):
    max_len_1 = max([s[0] for s in shapes])
    max_len_2 = max([s[1] for s in shapes])
    tensors = [torch.ones(shape + trailing_dims) for shape in shapes]
    padded = pad_nd_tensor(
        tensors, num_padding_dimensions=2, batch_first=False, padding_value=0
    )
    assert padded.shape == (max_len_1, max_len_2, len(shapes)) + trailing_dims

    padded = pad_nd_tensor(
        tensors, num_padding_dimensions=2, batch_first=True, padding_value=0
    )
    assert padded.shape == (len(shapes), max_len_1, max_len_2) + trailing_dims


def test_pad_2d_sequence_throws():
    shapes = [(11,), (13,)]
    tensors = [torch.ones(shape) for shape in shapes]
    with pytest.raises(Exception) as e_info:
        pad_nd_tensor(tensors, num_padding_dimensions=2, batch_first=False)
    with pytest.raises(Exception) as e_info:
        pad_nd_tensor(tensors, num_padding_dimensions=2, batch_first=True)
