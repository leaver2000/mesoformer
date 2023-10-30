import itertools

import model.conv4d


def test_conv4d():
    kernel_size = (1, 1, 1, 1)
    padding = (0, 0, 0, 0)
    stride = (1, 1, 1, 1)
    dilation = (1, 1, 1, 1)
    shape = (Batch, _, l_i, d_i, h_i, w_i) = (10, 5, 8, 12, 40, 80)
    (l_k, d_k, h_k, w_k) = kernel_size
    (l_p, d_p, h_p, w_p) = padding
    (l_d, d_d, h_d, w_d) = dilation
    (l_s, d_s, h_s, w_s) = stride
    l_o = (l_i + 2 * l_p - (l_k) - (l_k - 1) * (l_d - 1)) // l_s + 1
    d_o = (d_i + 2 * d_p - (d_k) - (d_k - 1) * (d_d - 1)) // d_s + 1
    h_o = (h_i + 2 * h_p - (h_k) - (h_k - 1) * (h_d - 1)) // h_s + 1
    w_o = (w_i + 2 * w_p - (w_k) - (w_k - 1) * (w_d - 1)) // w_s + 1
    expect = l_o, d_o, h_o, w_o
    assert expect == tuple(
        itertools.starmap(
            model.conv4d._get_dimension_size,
            zip(shape[2:], kernel_size, padding, dilation, stride),
        )
    )
