import numpy as np
import tensorflow as tf


"""
Disclaimer: the weight_variable_glorot function from this file comes
from the tkipf/gae original repository on Graph Autoencoders
"""


def weight_variable_glorot(input_dim, output_dim, name = ""):

    """
    Creates a weight variable with Glorot and Bengio's initialization
    """

    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval = - init_range,
                                maxval = init_range, dtype = tf.float32)

    return tf.Variable(initial, name = name)