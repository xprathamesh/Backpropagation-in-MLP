#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py
# Modified by Prathamesh

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    a = []
    activations = []
    activations.append(x)
    for layer in range(1, num_layers):
        a.append(np.dot(weightsT[layer-1], activations[layer-1]) + biases[layer-1])
        activations.append(sigmoid(a[layer-1]))
    ###

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)

    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    nabla_b[-1] = delta
    nabla_wT[-1] = np.dot(delta, activations[-2].transpose())
    for layer in range(2, num_layers):
        delta = np.dot(weightsT[-layer+1].transpose(), delta) * sigmoid_prime(a[-layer])
        nabla_b[-layer] = delta
        nabla_wT[-layer] = np.dot(delta, activations[-layer-1].transpose())
    ###

    return (nabla_b, nabla_wT)
