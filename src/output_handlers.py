import tensorflow as tf
import numpy as np
from load import categories, CATEGORIES

class _OutputHandler(object):
    """
    Class which contains all attributes specific to an output type
    """

    @property
    def output_dim(self):
        raise NotImplementedError('Output dimension not defined')

    def init_placeholders(self):
        raise NotImplementedError('Output handler placeholders not defined')

    def feed_dict(self, batch):
        raise NotImplementedError('Target feed_dict constructor not defined')

    @property
    def predictions(self):
        return self.outputs

    # Abstract method to create error metric
    @property
    def error(self):
        raise NotImplementedError('Error metric not defined')

    # Abstract method to create accuracy metric
    @property
    def accuracy(self):
        raise NotImplementedError('Accuracy metric not defined')

    @property
    def data_config(self):
        return {}


class WaveClasses(_OutputHandler):
    @property
    def output_dim(self):
        return 3

    def init_placeholders(self):
        self.targets = tf.placeholder(tf.int32, [None, self.output_dim], name='targets')

    def feed_dict(self, batch):
        # one-hot encode targets
        targets = np.zeros((len(batch), self.output_dim))
        targets[range(len(batch)), batch.id] = 1
        return {self.targets: targets}

    @property
    def predictions(self):
        return tf.nn.softmax(self.outputs)

    @property
    def error(self):
        # calculate the softmax cross entropy loss, this limits self.outputs to sum to 1
        return tf.losses.softmax_cross_entropy(self.targets, self.outputs)

    @property
    def accuracy(self):
        # calculate the number of samples which the label with the highest probability was correct
        return tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.targets, 1), tf.argmax(self.outputs, 1)),
            tf.float32), name='accuracy')


class EmoDimensional(_OutputHandler):
    """
    Handles specific settings for predicting emotion dimensions
    Average of annotator labels (not including self-evaluation) normalised to range [0, 1]
    """

    def __init__(self):
        self.emotion_set = ['val', 'act', 'dom']

    @property
    def output_dim(self):
        return len(self.emotion_set)

    def init_placeholders(self):
        self.targets = tf.placeholder(tf.float32, [None, self.output_dim], name='targets')

    def feed_dict(self, batch):
        return {self.targets: batch.target_dimensional}

    @property
    def predictions(self):
        return tf.sigmoid(self.outputs)

    @property
    def error(self):
        # calculate the sigmoid cross entropy loss, limits each dimension to range [0, 1]
        return tf.losses.sigmoid_cross_entropy(self.targets, self.outputs)

    @property
    def accuracy(self):
        # an accuracy metric is not applicable to emotion profiles
        return tf.constant(0., name='accuracy')


class EmoProfile(_OutputHandler):
    """
    Handles specific settings for predicting emotion profiles, i.e. probability distributions over emotion classes
    """

    def __init__(self, emotion_set=CATEGORIES):
        self.emotion_mask = np.in1d(CATEGORIES, emotion_set)

        # ensure that the emotion set is in the correct order by taking it from categories (not the method's parameter)
        self.emotion_set = CATEGORIES[self.emotion_mask]

    @property
    def output_dim(self):
        # ['Anger', 'Disgust', 'Excited', 'Fear', 'Frustration',
        #  'Happiness', 'Neutral', 'Other', 'Sadness', 'Surprise']
        return len(self.emotion_set)

    def init_placeholders(self):
        self.targets = tf.placeholder(tf.float32, [None, self.output_dim], name='targets')

    def feed_dict(self, batch):
        return {self.targets: batch.target_categorical_masked}

    @property
    def predictions(self):
        return tf.nn.softmax(self.outputs)

    @property
    def error(self):
        # calculate the softmax cross entropy loss, this limits self.outputs to sum to 1
        return tf.losses.softmax_cross_entropy(self.targets, self.outputs)

    @property
    def accuracy(self):
        # an accuracy metric is not applicable to emotion profiles
        return tf.constant(0., name='accuracy')

    @property
    def data_config(self):
        return {'emotion_mask_profile': self.emotion_mask}


class EmoProfileBasic4(EmoProfile):

    def __init__(self):
        super(EmoProfileBasic4, self).__init__(['Anger', 'Happiness', 'Neutral', 'Sadness'])


class EmoCategoricalConsensus(_OutputHandler):
    """
    Handles specific settings for predicting emotion label
    """

    def __init__(self, emotion_set=categories):
        self.emotion_mask = np.in1d(categories, emotion_set)

        # ensure that the emotion set is in the correct order by taking it from categories (not the method's parameter)
        self.emotion_set = categories[self.emotion_mask]

    @property
    def output_dim(self):
        # ['ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'sad', 'sur', 'oth', 'xxx']
        # 'xxx' means no consensus
        return len(self.emotion_set)

    def init_placeholders(self):
        self.targets = tf.placeholder(tf.float32, [None, self.output_dim], name='targets')

    def feed_dict(self, batch):
        return {self.targets: batch.target_one_of_k_cat_masked}

    @property
    def predictions(self):
        return tf.nn.softmax(self.outputs)

    @property
    def error(self):
        # calculate the softmax cross entropy loss, this limits self.outputs to sum to 1
        return tf.losses.softmax_cross_entropy(self.targets, self.outputs)

    @property
    def accuracy(self):
        # calculate the number of samples which the label with the highest probability was correct
        return tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.targets, 1)),
            tf.float32), name='accuracy')

    @property
    def data_config(self):
        return {'emotion_mask': self.emotion_mask}


class EmoCategoricalBasic4(EmoCategoricalConsensus):

    def __init__(self):
        super(EmoCategoricalBasic4, self).__init__(['ang', 'hap', 'neu', 'sad'])


class EmoCategoricalHappySad(EmoCategoricalConsensus):

    def __init__(self):
        super(EmoCategoricalHappySad, self).__init__(['hap', 'sad'])



