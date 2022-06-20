from modularity_aware_gae.initializations import weight_variable_glorot
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS
_LAYER_UIDS = {} # Global unique layer ID dictionary for layer name assignment


"""
Disclaimer: functions and classes from this file mainly come
from the tkipf/gae and deezer/gravity_graph_autoencoders repositories
"""


def get_layer_uid(layer_name = ''):

    """
    Helper function, assigns unique layer IDs
    """

    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):

    """
    Dropout for sparse tensors
    """

    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype = tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):

    """
    Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):

    """
    Graph convolution layer
    """

    def __init__(self, input_dim, output_dim, adj, dropout = 0., act = tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):

    """
    Graph convolution layer for sparse inputs
    """

    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout = 0., act = tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):

    """
    Inner product decoder layer
    """

    def __init__(self, fastgae, sampled_nodes, dropout = 0., act = tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.sampled_nodes = sampled_nodes # Nodes from sampled subgraph to decode
        self.fastgae = fastgae # Whether to use the FastGAE framework

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        # If FastGAE is used, we only reconstruct the sampled subgraph
        if self.fastgae:
            inputs = tf.gather(inputs, self.sampled_nodes)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class DistanceDecoder(Layer):

    """
    Exponential L2 distance term from the proposed
    modularity-inspired loss term
    """

    def __init__(self, fastgae, sampled_nodes, dropout = 0., act = tf.nn.sigmoid, **kwargs):
        super(DistanceDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.sampled_nodes = sampled_nodes # Nodes from sampled subgraph to decode
        self.fastgae = fastgae # Whether to use the FastGAE framework

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        if self.fastgae:
            inputs = tf.gather(inputs, self.sampled_nodes)
        # Get pairwise node distances in embedding
        dist = pairwise_distance(inputs)
        # Exponential
        outputs = tf.exp(- FLAGS.gamma * tf.reshape(dist, [-1]))
        outputs = self.act(outputs)
        return outputs


def pairwise_distance(X, eps = 0.1):

    """
    Pairwise distances between node pairs
    :param X: n*d embedding matrix
    :param epsilon: add a small value to distances for numerical stability
    :return: n*n matrix of squared Euclidean distances
    """

    x1 = tf.reduce_sum(X * X, 1, True)
    x2 = tf.matmul(X, tf.transpose(X))
    return x1 - 2 * x2 + tf.transpose(x1) + eps