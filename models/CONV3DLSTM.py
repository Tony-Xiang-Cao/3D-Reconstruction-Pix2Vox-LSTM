# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 22:46:11 2021

@author: caoxi
"""

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv, conv3d2d, sigmoid
from theano.tensor.signal import pool
import torch

class Conv3DLSTM(torch.nn.Module):
    """Convolution 3D LSTM module
    Unlike a standard LSTM cell witch doesn't have a spatial information,
    Convolutional 3D LSTM has limited connection that respects spatial
    configuration of LSTM cells.
    The filter_shape defines the size of neighbor that the 3D LSTM cells will consider.
    """

    def __init__(self, prev_layer, filter_shape, padding=None, params=None):

        super().__init__(prev_layer)
        prev_layer._input_shape
        n_c = filter_shape[0]
        n_x = self._input_shape[2]
        n_neighbor_d = filter_shape[1]
        n_neighbor_h = filter_shape[2]
        n_neighbor_w = filter_shape[3]

        # Compute all gates in one convolution
        self._gate_filter_shape = [4 * n_c, 1, n_x + n_c, 1, 1]

        self._filter_shape = [filter_shape[0],  # num out hidden representation
                              filter_shape[1],  # time
                              self._input_shape[2],  # in channel
                              filter_shape[2],  # height
                              filter_shape[3]]  # width
        self._padding = padding

        # signals: (batch,       in channel, depth_i, row_i, column_i)
        # filters: (out channel, in channel, depth_f, row_f, column_f)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        if params is None:
            #self.W = Weight(self._filter_shape, is_bias=False)
            #self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1, filler='constant')
            params = [self.W, self.b]
        else:
            self.W = params[0]
            self.b = params[1]

        self.params = [self.W, self.b]

        if padding is None:
            self._padding = [0, int((filter_shape[1] - 1) / 2), 0, int((filter_shape[2] - 1) / 2),
                             int((filter_shape[3] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                    input_shape[0],
                                    input_shape[1] + 2 * padding[1],
                                    input_shape[2],
                                    input_shape[3] + 2 * padding[3],
                                    input_shape[4] + 2 * padding[4])

        padded_input = tensor.set_subtensor(padded_input[:, padding[1]:padding[1] + input_shape[
            1], :, padding[3]:padding[3] + input_shape[3], padding[4]:padding[4] + input_shape[4]],
                                            self._prev_layer.output)

        self._output = conv3d2d.conv3d(padded_input, self.W.val) + \
            self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')