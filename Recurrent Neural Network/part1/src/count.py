#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: count.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from src.lstm import LSTMcell
import src.assign as assign
import matplotlib.pyplot as plt

def count_0_in_seq(input_seq, count_type):
    """ count number of digit '0' in input_seq

    Args:
        input_seq (list): input sequence encoded as one hot
            vectors with shape [num_digits, 10].
        count_type (str): type of task for counting.
            'task1': Count number of all the '0' in the sequence.
            'task2': Count number of '0' after the first '2' in the sequence.
            'task3': Count number of '0' after '2' but erase by '3'.

    Return:
        counts (int)
    """

    if count_type == 'task1':
        # Count number of all the '0' in the sequence.
        # create LSTM cell
        cell = LSTMcell(in_dim=10, out_dim=1)
        # assign parameters
        assign.assign_weight_count_all_0_case_1(cell, in_dim=10, out_dim=1)
        # initial the first state
        prev_state = [0.]
        g = i = o = f = 0
        # read input sequence one by one to count the digits
        for idx, d in enumerate(input_seq):
            prev_state,g,i,o,f = cell.run_step([d], prev_state=prev_state)
        count_num = int(np.squeeze(prev_state))
        return count_num

    if count_type == 'task2':
        # Count number of '0' after the first '2' in the sequence.
        in_dimension = (len(input_seq),2)
        cell = LSTMcell(in_dim=10, out_dim=2)
        assign.assign_weight_count_all_case_2(cell, in_dim=10, out_dim=2)
        prev_state = np.zeros((len(input_seq) + 1, 2))
        g = np.zeros(in_dimension)
        i = np.zeros(in_dimension)
        f = np.zeros(in_dimension)
        o = np.zeros(in_dimension)
        # read input sequence one by one to count the digits
        for idx, d in enumerate(input_seq):
            prev_state[idx + 1], g[idx], i[idx], f[idx], o[idx] = cell.run_step([d], prev_state=prev_state[idx])
        i_gate = (g[:, 0] * i[:, 0]).astype(int)
        inSequence = np.argmax((input_seq), axis=1)
        plt.bar(np.arange(len(inSequence)), inSequence)
        plt.xticks(np.arange(len(inSequence)), inSequence)
        plt.title('Input Sequence')
        plt.show()
        plt.plot(prev_state[:, 0], 'y-', g[:, 0], 'm^', i_gate, 'gP', f[:, 0], 'r*', o[:, 0], 'b.')
        plt.legend(['Internal state', 'Input node', 'Input gate', 'Forget gate', 'Output gate'])
        plt.xlabel("Time Step")
        plt.ylabel("Count")
        plt.title('Task 2')
        plt.show()
        count_num = int(prev_state[-1][0])
        return count_num

    if count_type == 'task3':
        # Count number of '0' in the sequence when receive '2', but erase
        # the counting when receive '3', and continue to count '0' from 0
        # until receive another '2'.
        in_dimension = (len(input_seq), 2)
        cell = LSTMcell(in_dim = 10, out_dim=2)
        assign.assign_weight_count_all_case_3(cell, in_dim=10, out_dim=2)
        prev_state = np.zeros((len(input_seq) + 1, 2))
        g = np.zeros(in_dimension)
        i = np.zeros(in_dimension)
        f = np.zeros(in_dimension)
        o = np.zeros(in_dimension)
        for idx, d in enumerate(input_seq):
            prev_state[idx + 1], g[idx], i[idx], f[idx], o[idx] = cell.run_step([d], prev_state=prev_state[idx])
        i_gate = (g[:, 0] * i[:, 0]).astype(int)
        plt.plot(prev_state[:, 0], 'y-', g[:, 0], 'm^', i_gate, 'gP', f[:, 0], 'r*', o[:, 0], 'b.')
        plt.legend(['Internal state', 'Input node', 'Input gate', 'Forget gate', 'Output gate'])
        plt.xlabel("Time Step")
        plt.ylabel("Count")
        plt.title('Task 3')
        plt.show()
        count_num = int(prev_state[-1][0])
        return count_num
