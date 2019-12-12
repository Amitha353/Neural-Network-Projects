#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lstm.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

SMALL_NUM = 1e-6


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def sigmoid(z):
    z = np.exp(-z)
    return 1 / (1 + z)


def identity(inputs):
    return inputs


class LSTMcell(object):
    """ LSTM cell class 

        Use the definition in:
        "A Critical Review of Recurrent Neural Networks for Sequence Learning"
    """

    def __init__(self, in_dim, out_dim, out_activation=identity):
        """ Initialize LSTM cell

        Args:
            in_dim (int): dimension of input
            out_dim (int): dimension of internal state and output
        """
        self._out_act = out_activation
        self.create_paramters(in_dim, out_dim)

    def run_step(self, inputs, prev_state):
        g = tanh(np.matmul(inputs, self.wgx) + np.matmul(prev_state, self.wgh) + self.bg)
        i = sigmoid(np.matmul(inputs, self.wix) + np.matmul(prev_state, self.wih) + self.bi)
        f = sigmoid(np.matmul(inputs, self.wfx) + np.matmul(prev_state, self.wfh) + self.bf)
        o = sigmoid(np.matmul(inputs, self.wox) + np.matmul(prev_state, self.woh) + self.bo)

        self.s = np.multiply(g, i) + np.multiply(self.s, f)
        return (np.multiply(self._out_act(self.s), o), g, i, f, o)

    def create_paramters(self, in_dim, out_dim):
        """ Initialize paramters for LSTM cell

        Args:
            in_dim (int): dimension of input
            out_dim (int): dimension of internal state and output
        """
        # internal state
        self.s = np.zeros((1, out_dim))

        # parameters for input node g
        # wgx - weight for input x
        # wgh - weight for previous state
        # bg - biases
        self.wgx = np.zeros((in_dim, out_dim))
        self.wgh = np.zeros((out_dim, out_dim))
        self.bg = np.zeros((1, out_dim))

        # parameters for input gate i
        self.wix = np.zeros((in_dim, out_dim))
        self.wih = np.zeros((out_dim, out_dim))
        self.bi = np.zeros((1, out_dim))

        # parameters for forget gate f
        self.wfx = np.zeros((in_dim, out_dim))
        self.wfh = np.zeros((out_dim, out_dim))
        self.bf = np.zeros((1, out_dim))

        # parameters for output gate o
        self.wox = np.zeros((in_dim, out_dim))
        self.woh = np.zeros((out_dim, out_dim))
        self.bo = np.zeros((1, out_dim))

    def set_config_by_name(self, name, val):
        """ Set parameter values by the dictionary name

        Args:
            name (string): key of dictionary
        """
        setattr(self, name, val)
