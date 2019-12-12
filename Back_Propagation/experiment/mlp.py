#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def show_metric_plot(training_dataset, evaluation_dataset, epochs, label1, label2, y_label, plt_title):
    epochs = list(range(epochs))
    plt.plot(epochs, training_dataset, 'b-', epochs, evaluation_dataset, 'r-')
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.legend((label1, label2))
    plt.show()

def main(phase):
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # pdb.set_trace()
    # construct the network
    model = network2.Network([784, 20, 10])
    # train the network using SGD
    if phase == "train":
        result = model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
        show_metric_plot(result[2], result[0],100, 'Training Error', 'Validation Error', 'Loss', 'Loss v/s Epochs')
        evaluation_accuracy = list(map(lambda x: x * 100 / 10000, result[1]))
        training_accuracy = list(map(lambda x: x * 100 / 3000, result[3]))
        show_metric_plot(training_accuracy, evaluation_accuracy,100, 'Training Accuracy', 'Validation Accuracy', 'Accuracy %', 'Accuracy v/s Epochs')
        model.save('train_model')
    if phase == "test":
        net = network2.load('train_model')
        print('Accuracy for Test Data:{}%'.format(net.accuracy(test_data) / 100))

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main("train")
    if FLAGS.gradient:
        gradient_check()
    if FLAGS.test:
        main("test")
