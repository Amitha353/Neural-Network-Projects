#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: assgin.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


def assign_weight_count_all_0_case_1(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence

    Input node only receives digit '0' and all the gates are
    always open.

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    param_dict['wgx'] = [[100.] if i == 0 else [0.] for i in range(10)]
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    param_dict['wix'] = np.zeros((in_dim, out_dim))
    param_dict['wih'] = np.zeros((out_dim, out_dim))
    param_dict['bi'] = 100. * np.ones((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])


def assign_weight_count_all_case_2(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence

    Input node receives all the digits '0' but input gate only
    opens for digit '0'. Other gates are always open.

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}

    wgx = np.zeros(shape=(10, 2))
    wgx[0, 0] = 100  # for 0 in first column
    wgx[2, 1] = 100  # for 2 in second column
    param_dict['wgx'] = wgx
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    wix = -100. * np.ones(shape=(10, 2))
    wix[2, 1] = 100  # mask for 2
    param_dict['wix'] = wix
    wih = np.zeros((out_dim, out_dim))
    wih[0] = wih[1] = [200, 200]
    param_dict['wih'] = wih
    param_dict['bi'] = np.zeros((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))  # always ON

    # o(t) - Output gate
    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))  # always ON

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

    return param_dict


# Task 3
def assign_weight_count_all_case_3(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence

        Input node receives all the digits '0' but input gate only
        opens for digit '0'. Forget gate clears after the third 0.
        The output gate is always open

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}

    # g(t) - input node
    wgx = np.zeros(shape=(10, 2))
    wgx[0, 0] = 100  # for 0 in first column
    wgx[2, 1] = 100  # for 2 in second column
    param_dict['wgx'] = wgx
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    # i(t) - Input gate
    wix = -100. * np.ones(shape=(10, 2))
    wix[2, 1] = 100  # mask for 2
    param_dict['wix'] = wix
    wih = np.zeros((out_dim, out_dim))
    wih[0] = wih[1] = [200, 200]
    param_dict['wih'] = wih
    param_dict['bi'] = np.zeros((1, out_dim))

    # f(t) - Forget gate
    wfx = 100. * np.ones(shape=(10, 2))
    wfx[3, 0] = -100  # forget/clear at 3
    wfx[3, 1] = -100  # forget/clear at 3
    param_dict['wfx'] = wfx
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = np.zeros((1, out_dim))

    # o(t) - Output gate
    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))  # always ON

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

    return param_dict
    wih[0] = wih[1] = [200, 200]
    param_dict['wih'] = wih
    param_dict['bi'] = np.zeros((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))  # always ON

    # o(t) - Output gate
    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))  # always ON

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

    return param_dict


# Task 3
def assign_weight_count_all_case_3(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence

        Input node receives all the digits '0' but input gate only
        opens for digit '0'. Forget gate clears after the third 0.
        The output gate is always open

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}

    # g(t) - input node
    wgx = np.zeros(shape=(10, 2))
    wgx[0, 0] = 100  # for 0 in first column
    wgx[2, 1] = 100  # for 2 in second column
    param_dict['wgx'] = wgx
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    # i(t) - Input gate
    wix = -100. * np.ones(shape=(10, 2))
    wix[2, 1] = 100  # mask for 2
    param_dict['wix'] = wix
    wih = np.zeros((out_dim, out_dim))
    wih[0] = wih[1] = [200, 200]
    param_dict['wih'] = wih
    param_dict['bi'] = np.zeros((1, out_dim))

    # f(t) - Forget gate
    wfx = 100. * np.ones(shape=(10, 2))
    wfx[3, 0] = -100  # forget/clear at 3
    wfx[3, 1] = -100  # forget/clear at 3
    param_dict['wfx'] = wfx
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = np.zeros((1, out_dim))

    # o(t) - Output gate
    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))  # always ON

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

    return param_dict