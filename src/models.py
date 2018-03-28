from collections import defaultdict
from helper import save_to_file, variable_summaries, FC_layer
import tensorflow as tf
import numpy as np
import os


class _Model(object):
    
    def __init__(self, experiment_name,
                 use_cross_validation=False, cross_validation_fold=None,
                 add_summaries=False, ckpt_interval=0, verbose=True, **kwargs):
        
        self.experiment_name = experiment_name
        self.use_cross_validation = use_cross_validation
        self.cross_validation_fold = cross_validation_fold
        self.add_summaries = add_summaries
        self.ckpt_interval = ckpt_interval
        self.verbose = verbose
        
        tf.reset_default_graph()

        # add handlers from kwargs to self (also initialising input/output handlers)
        with tf.variable_scope('placeholders'):
            self.add_handlers(**kwargs)

        # initialise placeholders
        self.init_placeholders(**kwargs)

        # load the data provider
        self.load_provider(**kwargs)

        # build the computational graph
        with tf.variable_scope('layers'):
            self.build_graph(**kwargs)

        # add prediction summaries
        if self.add_summaries:
            with tf.name_scope('predictions'):
                self.add_prediction_summaries(**kwargs)

        # add metrics and their summaries
        with tf.name_scope('metrics'):
            self.add_metrics(**kwargs)

        # add train step
        self.train_step = tf.train.AdamOptimizer().minimize(tf.losses.get_total_loss())

        # create session and init variables
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # create logging objects
        self.train_summaries = tf.summary.merge_all(key='train')
        self.valid_summaries = tf.summary.merge_all(key='valid')
        self.test_summaries = tf.summary.merge_all(key='test')
        self.summary_writer = tf.summary.FileWriter(
            os.path.join(os.environ['project_DIR'], 'results', self.experiment_name), self.sess.graph)
        self.saver = tf.train.Saver()
        self.log_data = {
            'epochs': [],
            'metrics': {
                output_name: {
                    metric: []
                    for metric in ['error_train', 'accuracy_train', 'error_valid', 'accuracy_valid', 'error_test', 'accuracy_test']
                }
                for output_name in self.output_handlers
            }
        }

    def __str__(self):
        return 'experiment_name: {}'.format(self.experiment_name)

    def add_handlers(self, **kwargs):
        raise NotImplementedError('add_handlers not defined in Model subclass {}'.format(self.__class__.__name__))

    def init_placeholders(self, **kwargs):
        raise NotImplementedError('init_placeholders not defined in Model subclass {}'.format(self.__class__.__name__))

    def load_provider(self, **kwargs):
        raise NotImplementedError('load_provider not defined in Model subclass {}'.format(self.__class__.__name__))

    def build_graph(self, **kwargs):
        raise NotImplementedError('build_graph not defined in Model subclass {}'.format(self.__class__.__name__))

    def add_prediction_summaries(self, **kwargs):
        raise NotImplementedError('add_predictions not defined in Model subclass {}'.format(self.__class__.__name__))

    def add_metrics(self, **kwargs):
        raise NotImplementedError('add_metrics not defined in Model subclass {}'.format(self.__class__.__name__))

    def __del__(self):
        if 'sess' in self.__dict__:
            self.sess.close()

    def train(self, num_epochs=40, begin_at=1):
        for e in range(begin_at, num_epochs+1):
            train_error, train_accuracy = self.train_epoch(e)
            valid_error, valid_accuracy = self.valid_epoch(e)
            # test_error, test_accuracy = self.test_epoch(e)

            if self.ckpt_interval and e % self.ckpt_interval == 0 and e != num_epochs:
                self.save_model(e)

            if self.verbose:
                print('epoch {0:2} error_train ........ acc_train ........ error_valid ........ acc_valid ........'
                      .format(e))
                for output_name in self.output_handlers.keys():
                    print('         {0:11} {1:.6f}           {2:.6f}             {3:.6f}           {4:.6f}'.format(
                        output_name[:11],
                        train_error[output_name], train_accuracy[output_name],
                        valid_error[output_name], valid_accuracy[output_name]))

        self.save_model()
        save_to_file(self.log_data, os.path.join(os.environ['project_DIR'], 'results', self.experiment_name, 'results.log'))
        return self.log_data

    def train_epoch(self, epoch):
        # train over all minibatches in data
        train_error, train_accuracy = defaultdict(lambda: 0.), defaultdict(lambda: 0.)
        for i, batch in enumerate(self.data_provider.train_data):

            _, summary, batch_error, batch_accuracy = self.sess.run(self.train_op,
                                                                    feed_dict=self.feed_dict(batch, is_training=True))

            self.summary_writer.add_summary(summary, (epoch - 1) * self.data_provider.num_batches + i + 1)

            for output_name in self.output_handlers.keys():
                train_error[output_name] += batch_error[output_name]
                train_accuracy[output_name] += batch_accuracy[output_name]

        for output_name in self.output_handlers.keys():
            train_error[output_name] /= (i + 1)
            train_accuracy[output_name] /= (i + 1)

        self.write_log(epoch, 'train', train_error, train_accuracy)
        return train_error, train_accuracy

    def valid_epoch(self, epoch):
        # evaluate model over all validation data
        valid_error, valid_accuracy = defaultdict(lambda: 0.), defaultdict(lambda: 0.)
        for i, batch in enumerate(self.data_provider.valid_data):

            summary, batch_error, batch_accuracy = \
                self.sess.run(self.valid_op, feed_dict=self.feed_dict(batch))

            self.summary_writer.add_summary(summary, (epoch - 1) * self.data_provider.num_batches_valid + i + 1)

            for output_name in self.output_handlers.keys():
                valid_error[output_name] += batch_error[output_name]
                valid_accuracy[output_name] += batch_accuracy[output_name]

        for output_name in self.output_handlers.keys():
            valid_error[output_name] /= (i + 1)
            valid_accuracy[output_name] /= (i + 1)

        self.write_log(epoch, 'valid', valid_error, valid_accuracy)
        return valid_error, valid_accuracy

    def test_epoch(self, epoch):
        # evaluate model over all test data
        test_error, test_accuracy = defaultdict(lambda: 0.), defaultdict(lambda: 0.)
        for i, batch in enumerate(self.data_provider.test_data):

            summary, batch_error, batch_accuracy = self.sess.run(self.test_op,
                                                                 feed_dict=self.feed_dict(batch))

            self.summary_writer.add_summary(summary, (epoch - 1) * self.data_provider.num_batches_test + i + 1)

            for output_name in self.output_handlers.keys():
                test_error[output_name] += batch_error[output_name]
                test_accuracy[output_name] += batch_accuracy[output_name]

        for output_name in self.output_handlers.keys():
            test_error[output_name] /= (i + 1)
            test_accuracy[output_name] /= (i + 1)

        self.write_log(epoch, 'test', test_error, test_accuracy)
        return test_error, test_accuracy

    def run_op_with_batch(self, op, batch):
        return self.sess.run(op, feed_dict=self.feed_dict(batch))

    def run_op_with_samples(self, op, file_paths):
        batch = self.data_provider.load_data(file_paths)
        return self.run_op_with_batch(op, batch)

    def write_log(self, epoch, stage, error, accuracy):
        if stage == 'train':
            self.log_data['epochs'].append(epoch)
        for output_name in self.output_handlers.keys():
            self.log_data['metrics'][output_name]['error_{}'.format(stage)].append(error[output_name])
            self.log_data['metrics'][output_name]['accuracy_{}'.format(stage)].append(accuracy[output_name])

    def ckpt_path(self, epoch=None):
        _ckpt_dir = os.path.join(os.environ['project_DIR'], 'results', '{}'.format(self.experiment_name), 'model')
        if not os.path.exists(_ckpt_dir):
            os.makedirs(_ckpt_dir)
        
        if epoch is None:
            _ckpt_path = os.path.join(_ckpt_dir, 'trained_model.ckpt')
        else:
            _ckpt_path = os.path.join(_ckpt_dir, 'epoch_{}.ckpt'.format(epoch))
            
        return _ckpt_path
    
    def save_model(self, epoch=None):
        save_path = self.ckpt_path(epoch)
        self.saver.save(self.sess, save_path)

        if self.verbose:
            print('Model saved to {}'.format(save_path))
    
    def restore_model(self, epoch=None):
        restore_path = self.ckpt_path(epoch)
        self.saver.restore(self.sess, restore_path)

        if self.verbose:
            print('Model restored from {}'.format(restore_path))
        
        return self

    # operations to run during training
    @property
    def train_op(self):
        return [self.train_step, self.train_summaries, self.error, self.accuracy]

    # operations to run during validation
    @property
    def valid_op(self):
        return [self.valid_summaries, self.error, self.accuracy]

    # operations to run during evaluation
    @property
    def test_op(self):
        return [self.test_summaries, self.error, self.accuracy]


class GraphModel(_Model):
    """
    Builds the model based on the graph specified in the setup config

    Does not currently support multiple input_handler's to one module_handler
    Does not currently support multiple module_handler's to one module_handler
    Need to implement a concatenation mechanism to all two handler's to be combined
    """

    def __init__(self, experiment_name, data_provider,
                 input_handlers, module_handlers, output_handlers, model_graph,
                 use_cross_validation=False, cross_validation_fold=None,
                 add_summaries=False, ckpt_interval=0, verbose=True):

        super(GraphModel, self).__init__(experiment_name,
                                         use_cross_validation, cross_validation_fold,
                                         add_summaries, ckpt_interval, verbose,
                                         data_provider=data_provider,
                                         input_handlers=input_handlers,
                                         module_handlers=module_handlers,
                                         output_handlers=output_handlers,
                                         model_graph=model_graph)

    def __str__(self):
        return 'experiment_name: {}\ninput_type: {}\noutput_type: {}\nmodule_handlers: {}\n'.format(
            self.experiment_name,
            ' '.join(input_handler.__class__.__name__ for input_handler in self.input_handlers.values()),
            ' '.join(output_handler.__class__.__name__ for output_handler in self.output_handlers.values()),
            ' '.join(map(str, self.module_handlers.values()))
        )

    # add handlers to self
    def add_handlers(self, input_handlers=None, module_handlers=None, output_handlers=None, **kwargs):
        self.input_handlers = {name: input_handler() for name, input_handler in input_handlers.iteritems()}
        self.module_handlers = module_handlers
        self.output_handlers = {name: output_handler() for name, output_handler in output_handlers.iteritems()}

    # placeholders
    def init_placeholders(self, **kwargs):
        for name, input_handler in self.input_handlers.iteritems():
            with tf.variable_scope(name):
                input_handler.init_placeholders()

        for name, module_handler in self.module_handlers.iteritems():
            with tf.variable_scope(name):
                module_handler.init_placeholders()

        for name, output_handler in self.output_handlers.iteritems():
            with tf.variable_scope(name):
                output_handler.init_placeholders()

    # load data provider
    def load_provider(self, data_provider=None, **kwargs):
        data_config = [('use_cross_validation', self.use_cross_validation),
                       ('cross_validation_fold', self.cross_validation_fold)]

        for input_handler in self.input_handlers.values():
            data_config += input_handler.data_config.items()

        for module_handler in self.module_handlers.values():
            data_config += module_handler.data_config.items()

        for output_handler in self.output_handlers.values():
            data_config += output_handler.data_config.items()

        self.data_provider = data_provider(dict(data_config))

    # build graph by iterating through nodes and accumulating the graph structure (in node_inputs)
    def build_graph(self, model_graph=None, **kwargs):
        node_inputs = defaultdict(list)
        graph_outputs = defaultdict(list)

        # traverse through graph, for each node check what type of handler it is before processing
        for node, children in model_graph:

            # set variables, node has an outgoing edge so cannot be an output_handler
            if node in self.input_handlers:
                handler = self.input_handlers[node]
                input_handler = handler
            elif node in self.module_handlers:
                handler = self.module_handlers[node]
                input_handler = handler.input_handler
            else:
                raise Exception('Graph must be structured such that only input_handlers and module_handlers '
                                'have outgoing edges, got\n{}'.format('\n'.join(map(str, model_graph))))

            # add graph pointers to current node
            handler.name = node
            handler.children = children

            # create outputs (using build_graph if node is a module_handler)
            if node in self.input_handlers:
                outputs = handler.inputs
            if node in self.module_handlers:
                with tf.variable_scope(node):
                    # build the graph using inputs collected from previous nodes
                    assert len(node_inputs[node]) == 1, (
                        'concatenation of handlers is not supported, ensure no handler has multiple incoming handlers'
                        '\n{}'.format(node_inputs[node]))
                    outputs = handler.build_graph(node_inputs[node][0], self.add_summaries)

            # prepare all children by collating their inputs and passing input_handler through the graph
            for next_node in children:
                # set variables, next_node has an incoming edge so cannot be an input_handler
                if next_node in self.module_handlers:
                    next_handler = self.module_handlers[next_node]
                elif next_node in self.output_handlers:
                    next_handler = self.output_handlers[next_node]
                else:
                    raise Exception('Graph must be structured such that only module_handlers and output_handlers '
                                    'have incoming edges, got\n{}'.format('\n'.join(map(str, model_graph))))

                # collect the previous node's outputs into node_inputs for use when we reach next_node
                node_inputs[next_node].append(outputs)  # only necessary if next_node is a module_handler

                # pass reference for input_handler to next_node for access to placeholders (e.g. is_training)
                next_handler.input_handler = input_handler

                if next_node in self.output_handlers:
                    graph_outputs[next_node].append(outputs)

        # add a linear fully-connected layer to the output handlers, according to their output dimension
        for node, output_handler in self.output_handlers.iteritems():
            with tf.variable_scope(node):
                assert len(graph_outputs[node]) == 1, (
                    'concatenation of handlers is not supported, ensure no handler has multiple incoming handlers'
                    '\n{}'.format(graph_outputs[node]))
                output_handler.outputs = FC_layer(graph_outputs[node][0], output_handler.output_dim, nonlinearity=tf.identity,
                                                  is_training=output_handler.input_handler.is_training, name='linear_layer')

        return self

    # add prediction summaries
    def add_prediction_summaries(self, **kwargs):
        for node, output_handler in self.output_handlers.iteritems():
            with tf.variable_scope(node):
                variable_summaries(output_handler.predictions)

    # add metrics and their summaries
    def add_metrics(self, **kwargs):
        self.error = {}
        self.accuracy = {}
        for output_name, output_handler in self.output_handlers.iteritems():
            with tf.variable_scope(output_name):
                error = output_handler.error
                accuracy = output_handler.accuracy

                tf.summary.scalar('error_train', error, collections=['train'])
                tf.summary.scalar('error_valid', error, collections=['valid'])
                tf.summary.scalar('error_test', error, collections=['test'])
                tf.summary.scalar('accuracy_train', accuracy, collections=['train'])
                tf.summary.scalar('accuracy_valid', accuracy, collections=['valid'])
                tf.summary.scalar('accuracy_test', accuracy, collections=['test'])

                self.error[output_name] = error
                self.accuracy[output_name] = accuracy
                tf.losses.add_loss(error)

    # populate feed_dict, this allows computation graph to access information from handlers
    def feed_dict(self, batch, is_training=False):
        items = []

        for input_handler in self.input_handlers.values():
            items += input_handler.feed_dict(batch, is_training).items()

        for module_handler in self.module_handlers.values():
            items += module_handler.feed_dict(batch).items()

        for output_handler in self.output_handlers.values():
            items += output_handler.feed_dict(batch).items()

        return dict(items)

    # iterate through batches of file_paths, calculating predictions and saving target values
    def predict(self, file_paths):
        predictions = {output_name: np.zeros((file_paths.shape[0], output_handler.output_dim))
                       for output_name, output_handler in self.output_handlers.iteritems()}
        targets = {output_name: np.zeros((file_paths.shape[0], output_handler.output_dim))
                   for output_name, output_handler in self.output_handlers.iteritems()}

        for i, batch in enumerate(self.data_provider.yield_batches(file_paths)):
            batch_size = len(batch)
            batch_slice = slice(batch_size * i,
                                batch_size * (i + 1))

            pred_dict = self.sess.run(self.prediction_op, feed_dict=self.feed_dict(batch))
            for name, (preds, trgts) in pred_dict.iteritems():
                predictions[name][batch_slice], targets[name][batch_slice] = preds, trgts

        return predictions, targets

    # Choose which operations to run for prediction
    @property
    def prediction_op(self):
        return {output_name: (output_handler.predictions, output_handler.targets)
                for output_name, output_handler in self.output_handlers.iteritems()}


class SimpleModel(GraphModel):
    """
    Builds a simple chain graph model: input -> modules -> output
    """

    def __init__(self, experiment_name, data_provider,
                 input_handler, module_handlers, output_handler,
                 use_cross_validation=False, cross_validation_fold=None,
                 add_summaries=False, ckpt_interval=0, verbose=True):

        input_handlers = {'input': input_handler}
        module_handlers = {'module-{}'.format(i+1): module_handler for i, module_handler in enumerate(module_handlers)}
        output_handlers = {'output': output_handler}

        model_graph = []
        for handler, next_handler in zip(input_handlers.keys() + module_handlers.keys(),
                                         module_handlers.keys() + output_handlers.keys()):
            model_graph.append((handler, [next_handler]))

        super(SimpleModel, self).__init__(experiment_name, data_provider,
                                          input_handlers, module_handlers, output_handlers, model_graph,
                                          use_cross_validation=use_cross_validation, cross_validation_fold=cross_validation_fold,
                                          add_summaries=add_summaries, ckpt_interval=ckpt_interval, verbose=verbose)



