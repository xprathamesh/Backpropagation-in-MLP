#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>
# Modified by Prathamesh

import sys
import argparse
import numpy as np
import pandas as pd
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

def export_predictions_to_csv(data, model):
    res = [(np.argmax(model.feedforward(x)), y) for (x,y) in zip(*data)]
    actual_values = []
    for x in res:
        actual_values.append(x[0])

    actual_values = np.array(actual_values)

    onehot = np.max(actual_values)+1
    onehot = np.eye(onehot)[actual_values]

    final = pd.DataFrame(onehot)
    final.to_csv("Predictions.csv", header=False, index=False)

def plot_this(epochs, training_cost, evaluation_cost, training_accuracy, evaluation_accuracy, train_nums, valid_nums):
    xvals = range(1, epochs+1)
    plt.figure()
    plt.plot(xvals, np.asarray(training_accuracy)/train_nums, 'c-')
    plt.plot(xvals, np.asarray(evaluation_accuracy)/valid_nums, 'm-')
    plt.legend(('Training Accuracy','Validation Accuracy'))
    plt.title('Training and Validation Accuracy')
    plt.savefig('train_val_acc.png')
    plt.figure()
    plt.plot(xvals, np.asarray(training_cost), 'c-')
    plt.plot(xvals, np.asarray(evaluation_cost), 'm-')
    plt.legend(('Training Loss','Validation Loss'))
    plt.title('Training and Validation Loss')
    plt.savefig('train_val_loss.png')

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 150, 10])
    # train the network using SGD
    epochs = 70
    mbsize = 64
    eta    = 1e-3
    lmbda  = 0.0
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = model.SGD(
        training_data   = train_data,
        epochs          = epochs,
        mini_batch_size = mbsize,
        eta             = eta,
        lmbda           = lmbda,
        evaluation_data = valid_data,
        monitor_evaluation_cost     = True,
        monitor_evaluation_accuracy = True,
        monitor_training_cost       = True,
        monitor_training_accuracy   = True)

    # model.save("2HNetwork.Data")

    plot_this(epochs, training_cost, evaluation_cost, training_accuracy, evaluation_accuracy, len(train_data[0]), len(valid_data[0]))
    print("[Testing Accuracy]: {}".format(model.accuracy(test_data)/len(test_data[0])))
    export_predictions_to_csv(test_data, model)

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
