import tensorflow as tf
from helper import variable_summaries, colorize, FC_layer, recurrent_cell
import numpy as np


class _ModuleHandler(object):
    """
    Class which builds a specific model type
    """

    def __init__(self):
        self.name = ''
        self.children = []

    def build_graph(self, inputs, add_summaries=False):
        raise NotImplementedError('Computation graph not defined')

    def init_placeholders(self):
        pass

    @property
    def data_config(self):
        return {}

    def feed_dict(self, batch):
        return {}


class FCModule(_ModuleHandler):
    """
    Implements build_graph function to create fully connected layers
    """

    def __init__(self, layer_dims=None, nonlinearity=None, dropout_prob=0.):

        super(FCModule, self).__init__()

        self.layer_dims = layer_dims
        self.nonlinearity = nonlinearity
        self.dropout_prob = dropout_prob

    def __repr__(self):
        return '{}-{}-({})'.format(self.__class__.__name__, self.nonlinearity.__name__,
                                   '_'.join(map(str, self.layer_dims)))

    def build_graph(self, inputs, add_summaries=False):
        prev_layer = inputs
        for i, layer_dim in enumerate(self.layer_dims):

            prev_layer = FC_layer(prev_layer, layer_dim, nonlinearity=self.nonlinearity,
                                  is_training=self.input_handler.is_training, dropout_prob=self.dropout_prob,
                                  name='layer_{}'.format(i + 1))
            if add_summaries: variable_summaries(prev_layer)

        return prev_layer


class CNNModule(_ModuleHandler):
    """
    Implements build_graph function to create N convolutional layers
    """

    def __init__(self,
                 num_filter_channels=None, filter_dims=None,
                 maxpooling=True, pool_dims=None, pool_strides=None,
                 nonlinearity=None):
        super(CNNModule, self).__init__()

        self.num_filter_channels = num_filter_channels
        self.filter_dims = filter_dims
        self.maxpooling = maxpooling
        self.pool_dims = pool_dims
        self.pool_strides = pool_strides
        self.nonlinearity = nonlinearity

    def __repr__(self):
        return '{}-{}-({}, {})'.format(self.__class__.__name__, self.nonlinearity.__name__,
                                       self.num_filter_channels, '_'.join(map(str, self.filter_dims)))

    def build_graph(self, inputs, add_summaries=False):
        prev_layer = tf.expand_dims(inputs, 3)
        for i, (num_filters, filter_dim) in enumerate(zip(self.num_filter_channels, self.filter_dims)):
            prev_layer = tf.layers.conv2d(prev_layer,
                                          num_filters, filter_dim,
                                          activation=self.nonlinearity,
                                          name='CNN_layer{}'.format(i + 1))
            if add_summaries: variable_summaries(prev_layer)

            if self.maxpooling:
                prev_layer = tf.layers.max_pooling2d(prev_layer,
                                                     self.pool_dims[i], self.pool_strides[i],
                                                     name='Pool_layer{}'.format(i + 1))

        flattened = tf.reshape(prev_layer, [-1, tf.reduce_prod(prev_layer.get_shape()[1:])])

        return flattened


class RNNModule(_ModuleHandler):
    """
    Implements build_graph function to create (stacked) recurrent layers
    """

    def __init__(self, layer_dims=None, cell_type=None, dropout_prob=0.):
        super(RNNModule, self).__init__()

        self.layer_dims = layer_dims
        self.cell_type = cell_type
        self.dropout_prob = dropout_prob

    def __repr__(self):
        return '{}-{}-({})'.format(self.__class__.__name__, self.cell_type.__name__,
                                   '_'.join(map(str, self.layer_dims)))

    def build_graph(self, inputs, add_summaries=False):
        # build list of recurrent cells
        cells = []
        for i, layer_dim in enumerate(self.layer_dims):
            cell = recurrent_cell(self.cell_type, layer_dim,
                                  self.input_handler.is_training, self.dropout_prob, 'RNN_cell{}'.format(i + 1))
            cells.append(cell)

        # create multilayer recurrent cell
        with tf.name_scope('multilayer_cell'):
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # create the actual recurrent architecture
        with tf.name_scope('outputs'):
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs,
                sequence_length=self.input_handler.sequence_lengths,
                dtype=tf.float32
            )
            if add_summaries: variable_summaries(outputs)

        # return outputs  # for seq2seq
        return outputs[:, -1, :]
        # return np.array(last_states[-1])  # same as above



