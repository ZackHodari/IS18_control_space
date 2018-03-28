import tensorflow as tf
from matplotlib.cm import get_cmap
import numpy as np
from tensorflow.python.ops import init_ops
import pickle
import os
import sys
from __init__ import _PROJECT_NAME


def get_task():
    project_DIR = os.path.join(os.path.abspath(__file__).split('/'+_PROJECT_NAME)[0], _PROJECT_NAME)
    sys.path.append(project_DIR)
    from egs.setup import tasks

    # retrieve the task set and output type from command line argument
    task_set = sys.argv[1]  # baseline_FC | baseline_RNN | tests
    assert task_set in tasks.keys(), (
        'task_set must be a key in tasks, got {}'.format(task_set))

    # set task_id and get task definition, using array job id or command line argument
    if 'SGE_TASK_ID' in os.environ and os.environ['SGE_TASK_ID'] != 'undefined':
        task_id = int(os.environ['SGE_TASK_ID'])
    else:
        task_id = int(sys.argv[2])

    assert 0 < task_id <= len(tasks[task_set]), (
        'task_id must be between 1 and {}, got {}'.format(len(tasks[task_set]), task_id))

    # get the individual task to be run
    task = tasks[task_set][task_id - 1]
    print('\ntask_set: {}\ntask_id: {}\ntask_name: {}\n'.format(task_set, task_id, task['name']))

    return task


# save/load functions that can easily be changed to different module backends
def save_to_file(data, filepath, ext='.pkl'):
    filepath += ext

    if ext in ['.cat', '.dim', '.eGeMAPS']:
        with open(filepath, 'wb') as f:
            data.tofile(f)
        return

    with open(filepath, 'w') as f:
        if ext == '.pkl':
            pickle.dump(data, f)
        elif ext == '.scp':
            f.write('\n'.join(data)+'\n')
        else:
            raise ValueError('Extension {} not supported'.format(ext))


def load_from_file(filepath, ext='.pkl'):
    filepath += ext

    if ext in ['.cat', '.dim', '.eGeMAPS']:
        with open(filepath, 'rb') as f:
            return np.fromfile(f, dtype=np.float32)

    with open(filepath, 'r') as f:
        if ext == '.pkl':
            data = pickle.load(f)
        elif ext == '.scp':
            data = map(str.strip, f.readlines())
        else:
            raise ValueError('Extension {} not supported'.format(ext))

    return data


def print_log(experiment_name):
    project_DIR = os.path.join(os.path.abspath(__file__).split('/'+_PROJECT_NAME)[0], _PROJECT_NAME)
    log_data = load_from_file(os.path.join(project_DIR, 'results', experiment_name, 'results.log'), '.pkl')

    # print('\naverage performance of model over 5 folds\n')
    for i, e in enumerate(log_data['epochs']):
        print('epoch {0:2} error_train ........ acc_train ........ error_valid ........ acc_valid ........'.format(e))
        for output_name, stats in log_data['metrics'].iteritems():
            print('         {0:11} {1:.6f}           {2:.6f}             {3:.6f}           {4:.6f}'.format(
                output_name[:11],  # up to 11 characters long
                stats['error_train'][i],
                stats['accuracy_train'][i],
                stats['error_valid'][i],
                stats['accuracy_valid'][i]))


def average_logs(log_data_list):
    output_handlers = list(set.union(*[set(log_data['metrics'].keys()) for log_data in log_data_list]))
    avg_log_data = {
        'epochs': log_data_list[0]['epochs'],
        'metrics': {
            output_name: {
                metric: np.mean(np.array([log_data['metrics'][output_name][metric] for log_data in log_data_list]), axis=0)
                for metric in ['error_train', 'accuracy_train', 'error_valid', 'accuracy_valid', 'error_test', 'accuracy_test']
            }
            for output_name in output_handlers
        }
    }

    return avg_log_data


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization)
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections=['train'])
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, collections=['train'])
        tf.summary.scalar('max', tf.reduce_max(var), collections=['train'])
        tf.summary.scalar('min', tf.reduce_min(var), collections=['train'])
        tf.summary.histogram('histogram', var, collections=['train'])


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b

    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value


def FC_layer(inputs, output_dim, nonlinearity=tf.nn.relu,
             is_training=None, dropout_prob=0., name='FC_layer'):

    if dropout_prob > 0.:
        inputs = tf.layers.dropout(inputs, dropout_prob, training=is_training)

    outputs = tf.layers.dense(inputs, output_dim, nonlinearity,
                              kernel_initializer=init_ops.glorot_normal_initializer(),
                              bias_initializer=init_ops.zeros_initializer(),
                              name=name)

    return outputs


def recurrent_cell(cell_type, output_dim,
                   is_training=None, dropout_prob=0., name='RNN_cell'):

    with tf.variable_scope(name):
        with tf.name_scope('basic_cell'):
            cell = cell_type(num_units=output_dim)

        with tf.name_scope('dropout_cell'):
            dropout_enabled = tf.logical_and(is_training, tf.greater(dropout_prob, 0.))
            keep_prob = tf.cond(dropout_enabled, lambda: 1. - tf.constant(dropout_prob), lambda: tf.constant(1.))
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)

    return cell



