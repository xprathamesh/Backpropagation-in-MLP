#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py
# Modified by Prathamesh

import numpy as np

def sigmoid(z):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
