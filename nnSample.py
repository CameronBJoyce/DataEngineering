    # -*- coding: utf-8 -*-
"""
Neural Network Classifier
"""

__author__ = """
### Author 1
Cameron Joyce
"""

__all__ = []

# **************************************************

import datetime
import numpy as np
import os.path
import sys
import tensorflow as tf
from types import SimpleNamespace

import ConversationFunctions as cf
from EarlyStop import EarlyStop
import LogUtilities as lu
import OutputClasses
from Timer import Timer

# **************************************************

MODEL_FILE_BASE = "AntexTrainedModel"
N_CLASSES = 2
SUMMARY_INTERVAL = 10

# **************************************************

class Classifier(object):
    """Neural Network Classifier

    This class defines the neural network model allowing it to be:

    - Trained
    - Saved/Restored
    - Used for inference


    Class is structured as a context manager to start and stop the
    Tensorflow session.
    """

    # **********************************************

    def __init__(self,
                 config,
                 features):
        """Setup the Tensorflow session and default graph

        Parameters
        ----------
        config : Config object
            Configuration parameters for the classifier (config.classifier)
        features : Features object
            Classifier feature metadata
        """

        self._config = config
        self._features = features
        self._batch_no = 0

    # **************************************************

    def __enter__(self):
        """Open the Tensorflow session

        Starts context managers for the tensorflow graph and session

        Returns
        -------
        Classifier Instance
            This object
        """

        self._graph_cm = tf.Graph().as_default()
        self._graph = self._graph_cm.__enter__()

        tf_config = tf.compat.v1.ConfigProto(log_device_placement=(self._config.tf_log))
        self._session = tf.compat.v1.Session(config=tf_config)
        self._session.__enter__()

        return self

    # **************************************************

    def __exit__(self,
                 exc_type,
                 exc_value,
                 exc_traceback):
        """Close the Tensorflow session and default graph
        """

        self._session.__exit__(exc_type, exc_value, exc_traceback)
        self._graph_cm.__exit__(exc_type, exc_value, exc_traceback)

    # **************************************************

    def build_model(self):
        """Build the Tensorflow graph for the model.

        Model will include all training operations in addition to just the NN.
        """

        # Having accumulated all the layers, build the actual rnn

        def _finish_lstm_layers(x, xlen):

            cell = tf.contrib.rnn.MultiRNNCell(lstm_layers,
                                               state_is_tuple=True)

            initial_state = cell.zero_state(batch_sz, tf.float32)

            outputs, self._final_state = tf.contrib.rnn.static_rnn(cell,
                                                                   x,
                                                                   sequence_length=xlen,
                                                                   initial_state=initial_state)

            return tf.reshape(tf.concat(outputs, 1), [-1, last_layer_sz], name='outputs_2d')

        # Create an lstm layer

        def _lstm_layer(sz, dropout_prob):

            layer = tf.nn.rnn_cell.LSTMCell(sz,
                                            forget_bias=0.0,
                                            state_is_tuple=True,
                                            reuse=tf.compat.v1.get_variable_scope().reuse)

            if dropout_prob is not None:
                layer = tf.contrib.rnn.DropoutWrapper(layer,
                                                      input_keep_prob=dropout_prob,
                                                      output_keep_prob=dropout_prob)

            return layer

        # Create a ReLU layer

        def _relu_layer(sz, inputs, i, dropout_prob):

            layer = tf.contrib.layers.fully_connected(inputs,
                                                      sz,
                                                      activation_fn=tf.nn.relu,
                                                      scope="ReLU_{}".format(i))

            if dropout_prob is not None:
                layer = tf.nn.dropout(layer, rate=1.0 - dropout_prob)

            return layer

        # Create a sigmoid layer

        def _sigmoid_layer(sz, inputs, i, dropout_prob):
            """
            Create a sigmoid layer
            """

            layer = tf.contrib.layers.fully_connected(inputs,
                                                      sz,
                                                      activation_fn=tf.sigmoid,
                                                      scope="ReLU_{}".format(i))

            if dropout_prob is not None:
                layer = tf.nn.dropout(layer, rate=1.0 - dropout_prob)

            return layer

        # Build the graph for the neural network itself

        n_steps = self._features.packetlimit

        initializer = tf.random_uniform_initializer(-self._config.parms.init_scaling,
                                                    self._config.parms.init_scaling)

        with tf.compat.v1.variable_scope("NeuralNetwork",
                                         reuse=None,
                                         initializer=initializer):

            # Setup the placeholders that will be input to the run

            self._x = tf.compat.v1.placeholder(tf.float32,
                                               [None, self._features.n_features, n_steps],
                                               name='x')

            self._xlen = tf.compat.v1.placeholder(tf.int32, [None], name='x_len')

            self._y = tf.compat.v1.placeholder(tf.int64, [None], name='y')

            if self._config.parms.dropout_keep_probability is not None:
                self._dropout_prob = tf.compat.v1.placeholder(tf.float32, name="dropout")
            else:
                self._dropout_prob = None

            self._learning_rate = tf.compat.v1.placeholder(tf.float32, name="learning_rate")

            # Loop building the layers

            x = [tf.squeeze(x_, [2], name="x_squeezed") for x_ in tf.split(self._x,
                                                                           n_steps,
                                                                           2)]

            batch_sz = tf.shape(x[0])[0]

            lstm_layers = []
            last_layer_type = 'l'

            for layer_no, layer in enumerate(self._config.topology):
                if layer[0] == 'l':
                    assert last_layer_type == 'l', "LSTM layers must preceed other layers"
                    lstm_layers.append(_lstm_layer(layer[1], self._dropout_prob))
                    last_layer_sz = layer[1]
                else:
                    if last_layer_type == 'l':
                        output = _finish_lstm_layers(x, self._xlen)
                    if layer[0] == 's':
                        _sigmoid_layer(layer[1], output, layer_no, self._dropout_prob)
                    elif layer[0] == 'r':
                        output = _relu_layer(layer[1], output, layer_no, self._dropout_prob)
                    else:
                        assert False, "Unknown layer type {}".format(layer[0])

                last_layer_type = layer[0]

            # If no non-lstm layers...finish up

            if last_layer_type == 'l':
                output = _finish_lstm_layers(x, self._xlen)

            # Add lstm summary information

            if self._config.tboard_dir is not None:
                with tf.name_scope("lstm_summaries"):
                    _add_lstm_summaries(len(lstm_layers))

            # Setup the output of the rnn layers and score the values

            softmax_w = tf.compat.v1.get_variable("Softmax_W", [output.shape[1], N_CLASSES])
            softmax_b = tf.compat.v1.get_variable("Softmax_B", [N_CLASSES])

            logits = tf.compat.v1.nn.xw_plus_b(output, softmax_w, softmax_b, name='logits')
            logits_3d = tf.compat.v1.reshape(logits,
                                             [batch_sz, n_steps, N_CLASSES],
                                             name='logits_3d')

            step_probs = tf.nn.softmax(logits_3d, name='predictions')

            batch_probs = tf.compat.v1.reduce_sum(step_probs, [1], name='batch_prob_sums')

            # batch_probs gives the probabilities for each class and should sum to 1 for each instancen
            
            self.batch_probs = tf.divide(batch_probs,
                                         tf.cast(n_steps, tf.float32),
                                         name='batch_probs')

        # Add output summary informatino

        if self._config.tboard_dir:
            with tf.name_scope("logit_summaries"):
                _var_summaries("Softmax w", softmax_w)
                _var_summaries("Softmax b", softmax_b)
                _var_summaries("Logits", logits_3d)

        # Compute the loss

        with tf.name_scope("Loss"):

            # Format y to the correct dimensions for the loss function

            y = tf.reshape(self._y, [-1, 1])
            y = tf.tile(y, [1, n_steps])

            loss_op = tf.contrib.seq2seq.sequence_loss(logits_3d,
                                                       y,
                                                       tf.ones([batch_sz, n_steps]))

            # So we get a name for the operation output

            self._loss_op = tf.identity(loss_op, name='loss_op')

            if self._config.tboard_dir:
                tf.summary.scalar("Training_Loss", self._loss_op)

        # Accuracy

        with tf.name_scope("Accuracy"):

            predictions = tf.argmax(self.batch_probs, 1, name='predictions')

            # Get the overall number of correct

            hits = tf.equal(predictions, self._y, name='hits')
            self._correct_op = tf.reduce_sum(tf.cast(hits, tf.int32), name='correct_op')

            # Get the details (only used for the test dataset)

            sum = tf.add(self._y, predictions)
            N = tf.reduce_sum(tf.cast(tf.equal(sum, 0), tf.int32), name="N")
            A = tf.reduce_sum(tf.cast(tf.equal(sum, 2), tf.int32), name="A")
            Y0 = tf.reduce_sum(tf.cast(tf.equal(self._y, 0), tf.int32), name="Y0")
            Y1 = tf.reduce_sum(tf.cast(tf.equal(self._y, 1), tf.int32), name="Y1")
            self._details_op = tf.stack([N, A, Y0, Y1], name="details_op")

        # Gradient descent training

        with tf.name_scope("Optimizer"):

            tvars = tf.compat.v1.trainable_variables()

            grads = tf.gradients(self._loss_op, tvars, name='get_gradients')

            if self._config.parms.gradient_clip_limit is not None:
                grads, _ = tf.clip_by_global_norm(grads, self._config.parms.gradient_clip_limit, name='clip_gradients')

            optimizer_class = getattr(tf.compat.v1.train,
                                      self._config.parms.optimizer.name)

            optimizer = optimizer_class(self._learning_rate,
                                        name='optimizer',
                                        **self._config.parms.optimizer.parms)

            self._training_op = optimizer.apply_gradients(zip(grads, tvars), name='training_op')

        # Add tensorboard summaries if requested

        if self._config.tboard_dir is not None:
            with tf.name_scope("final_summaries"):
                self._summary_op = tf.summary.merge_all()

    # **************************************************

    def infer_labeled(self,
                      convs):
        """Perform inference on a set of labeled conversations

        Each item in the data list will be dictionary containing:

        - The numpy array of features
        - The class label (if available)

        The results of the overall inference are summarized and printed.

        Parameters
        ----------
        convs : list
            List of conversations expanded as features

        Returns
        -------
        SimpleNamespace:
            Summary information on the conversations at the protocol level
        """

        # Run the test dataset

        with Timer() as inference_timer:
            inference_loss, inference_acc, inference_details = self._run_epoch(convs,
                                                                               [self._loss_op,
                                                                                self._correct_op,
                                                                                self._details_op])

        # Output test summary

        lu.info("Inference Summary:")
        lu.log_value("Inference", "loss: {:0.3f} accuracy: {:0.3f}%",
                     inference_loss,
                     inference_acc * 100)
        lu.log_value("Class 0 accuracy", "{}/{} ({:0.3f}%)",
                     inference_details[0],
                     inference_details[2],
                     inference_details[0] / inference_details[2] * 100.0)
        lu.log_value("Class 1 accuracy", "{}/{} ({:0.3f}%)",
                     inference_details[1],
                     inference_details[3],
                     inference_details[1] / inference_details[3] * 100.0 \
                     if inference_details[3] != 0 else 0.0)
        lu.log_value("Inference time", "{}", inference_timer())

        return SimpleNamespace(loss=float(inference_loss),
                               accuracy=float(inference_acc),
                               time=str(inference_timer()),
                               class_details=[SimpleNamespace(correct=inference_details[0],
                                                              total=inference_details[2]),
                                              SimpleNamespace(correct=inference_details[1],
                                                              total=inference_details[3])])

    # **************************************************

    def infer_unlabeled_from_generator(self,
                                       data_generator,
                                       summary):
        """Perform inference retrieving data from a generator.

        Each item in the data retrieved will be dictionary containing:

        - The numpy array of features

        The results of each instances inference is passed to the set of output
        classes loaded.

        This method should be used for unlabeled inference

        Parameters
        ----------
        data_generator : list generator
            Typically a generator function
        """

        # Show and initialize the output processors

        OutputClasses.log_classes("Output processors loaded:")
        output_processors = [cls() for cls in OutputClasses.get_classes()]
        [os.init() for os in output_processors]

        # Do the inference

        n_features = self._features.n_features
        max_steps = self._features.packetlimit
        x = np.empty([1, n_features, max_steps])

        total_convs = 0

        with Timer() as inference_timer:

            for conv in data_generator:

                features = conv['features']

                x.fill(0.0)
                x[0,
                  :,
                  0:min(max_steps, features.shape[0])] = conv['features'].T[:, 0:min(max_steps, features.shape[0])]
                xlen = [features.shape[0]]

                batch_probs = self._run_batch(x, xlen, None, self._batch_probs)

                # Pass the output as a tuple to each of the loaded output classes

                [oc([(batch_probs[0], conv['key'])]) for oc in output_processors]

                total_convs += 1

        # Get the summary information from the output processors

        summary.outputs = {op.__class__.__name__: op.term() for op in output_processors}
        summary.conversations_processed = total_convs

        lu.log_value("Inference time", "{}", inference_timer())

    # **************************************************

    def infer_unlabeled_from_list(self,
                                  convs):
        """Perform inference on a set of unlabeled conversations

        Each item in the data list will be dictionary containing:

        - The numpy array of features

        The results of the overall inference are summarized and printed.

        Parameters
        ----------
        convs : list
            List of conversations expanded as features

        Returns
        -------
        SimpleNamespace:
            Summary information on the conversations at the protocol level
        """

        # Show and initialize the output processors

        OutputClasses.log_classes("Output processors loaded:")
        output_processors = [cls() for cls in OutputClasses.get_classes()]
        [os.init(self._config) for os in output_processors]

        # Perform the inference

        out_vec = []

        with Timer() as inference_timer:
            for i, (x, xlen, y, keys) in enumerate(cf.get_feature_batch(convs,
                                                                        self._config.batch_size,
                                                                        self._features.n_features,
                                                                        self._features.packetlimit)):

                batch_probs = self._run_batch(x, xlen, None, self._batch_probs)

                for i, prob in enumerate(batch_probs):
                    out_vec.append((prob, keys[i]))

        # Pass the output to each of the loaded output classes

        [oc(out_vec) for oc in output_processors]

        # Get the summary information from the output processors

        op_dict = {op.__class__.__name__: op.term() for op in output_processors}

        # Print the total time and get summary info

        lu.log_value("Inference time", "{}", inference_timer())

        return SimpleNamespace(**dict(conversations_processed=len(out_vec)),
                               **op_dict)

    # **************************************************

    def init_model(self):
        """Initialize tensorflow variables
        """

        tf.compat.v1.global_variables_initializer().run()

    # **************************************************

    def load_model(self,
                   location):
        """Load a (hopefully) trained model.

        Parameters
        ----------
        location : string
            Directory from which to load the model
        """

        model_file_base = os.path.join(location, MODEL_FILE_BASE)

        # Restore the model

        restored_saver = tf.compat.v1.train.import_meta_graph(model_file_base + '.meta')
        restored_saver.restore(self._session, model_file_base)

        # Locate the operations we need

        g = self._session.graph

        self._loss_op = g.get_tensor_by_name('Loss/loss_op:0')
        self._correct_op = g.get_tensor_by_name('Accuracy/correct_op:0')
        self._details_op = g.get_tensor_by_name('Accuracy/details_op:0')
        self._batch_probs = g.get_tensor_by_name('NeuralNetwork/batch_probs:0')

        # Locate the placeholders

        self._x = g.get_tensor_by_name('NeuralNetwork/x:0')
        self._xlen = g.get_tensor_by_name('NeuralNetwork/x_len:0')
        self._y = g.get_tensor_by_name('NeuralNetwork/y:0')
        self._dropout_prob = g.get_tensor_by_name('NeuralNetwork/dropout:0')

    # **************************************************

    def print_model(self):
        """Display information about model variables
        """
        max_name = max(len(v.name) for v in tf.compat.v1.global_variables())
        max_shape = max(len(str(v.get_shape())) for v in tf.compat.v1.global_variables())
        max_vars = max(len(str(np.product(v.get_shape()))) for v in tf.compat.v1.global_variables())
        max_vars_fmt = len("{:,d}".format(10**max_vars))
        max_type = max(len(repr(v.dtype)) for v in tf.compat.v1.global_variables())

        hdr_str = "   {:{max_name}s} {:{max_shape}s} {:{max_vars_fmt}s} {:{max_type}s}"
        log_str = "   {:{max_name}s} {:{max_shape}s} {:{max_vars_fmt},d} {:{max_type}s}"

        lu.info("Tensorflow Variable Info:")
        lu.info(hdr_str.format("Name", "Shape", "Vars", "Type",
                               max_name=max_name,
                               max_shape=max_shape,
                               max_vars_fmt=max_vars_fmt,
                               max_type=max_type))

        lu.info(hdr_str.format('*' * max_name, '*' * max_shape, '*' * max_vars_fmt, '*' * max_type,
                               max_name=max_name,
                               max_shape=max_shape,
                               max_vars_fmt=max_vars_fmt,
                               max_type=max_type))

        for v in tf.compat.v1.global_variables():
            shape = v.get_shape()
            vars = int(np.product(shape))
            lu.info(log_str.format(v.name, str(shape), vars, repr(v.dtype),
                                   max_name=max_name,
                                   max_shape=max_shape,
                                   max_vars_fmt=max_vars_fmt,
                                   max_type=max_type))

        total_parms = sum(int(np.product(v.get_shape())) for v in tf.compat.v1.trainable_variables())
        lu.info(f"   Total Trainable Parameters: {total_parms}")

    # **************************************************

    def save_model(self,
                   location):
        """Save the generated model

        Parameters
        ----------
        location : string
            Directory in which to save the model
        """

        try:
            model_directory = os.path.join(location,
                                           MODEL_FILE_BASE)
            saver = tf.compat.v1.train.Saver()
            saver.save(self._session, model_directory)
            lu.info(f"Model saved to {location}")
        except (AttributeError, TypeError):
            lu.info("Unable to save model")

    # **************************************************

    def train_model(self,
                    training_data,
                    validation_data,
                    test_data):
        """Train the model

        Perform model training using the training and validation datasets
        according to the specifications in the configuration object.

        After training the test dataset (if present) if run.

        Parameters
        ----------
        training_data : list
            List of training instances (dict)
        validation_data : list
            List of validation instances (dict)
        test_data : list
            List of test instances (dict)

        returns
        -------
        Dict
            Training summary dictionary
        """

        training_start_time = datetime.datetime.now()

        summary = SimpleNamespace(epochs=[])

        if self._config.parms.early_stop.enabled:
            early_stopper = EarlyStop(self._config.parms.early_stop,
                                      summary.epochs)
        else:
            early_stopper = None

        for epoch in range(self._config.max_epochs):

            lu.info("Epoch {}:".format(epoch + 1))

            with Timer() as epoch_timer:

                lr = self._config.parms.learning_rate
                lr_decay = lr.decay ** max(epoch - lr.init_epochs, 0.0)
                learning_rate = lr.initial_rate * lr_decay

                train_loss, train_acc, _ = self._run_epoch(training_data,
                                                           [self._loss_op,
                                                            self._correct_op,
                                                            self._training_op],
                                                           random=True,
                                                           learning_rate=learning_rate)

                valid_loss, valid_acc, _ = self._run_epoch(validation_data,
                                                           [self._loss_op,
                                                            self._correct_op])

            elapsed_time = epoch_timer.end - training_start_time

            lu.log_value("Learning rate", "{:0.3f}", learning_rate)
            lu.log_value("Training", "loss: {:0.3f} accuracy: {:0.3f}%",
                         train_loss,
                         train_acc * 100)
            lu.log_value("Validation", "loss: {:0.3f} accuracy: {:0.3f}%",
                         valid_loss,
                         valid_acc * 100)
            lu.log_value("Epoch training time", "{}", epoch_timer())
            lu.log_value("Elapsed training time", "{}", elapsed_time)
            lu.log_value("Time remaining", "{}",
                         elapsed_time / (epoch + 1) \
                         * (self._config.max_epochs - (epoch + 1)))

            summary.epochs.append(dict(learning_rate=float(learning_rate),
                                       training_loss=float(train_loss),
                                       training_acc=float(train_acc),
                                       validation_loss=float(valid_loss),
                                       validation_acc=float(valid_acc),
                                       elapsed_time=str(elapsed_time),
                                       epoch_time=str(epoch_timer())))

            # Early stop testing

            if early_stopper and early_stopper(epoch, valid_loss):
                break

        # Output training summary

        lu.info("Training Summary:")
        lu.log_value("Training time", "{}", elapsed_time)
        lu.log_value("Training epochs", "{}", epoch + 1)

        summary.training_summary = dict(time=str(elapsed_time),
                                        epochs=epoch + 1)

        # Run the test dataset

        with Timer() as test_timer:
            test_loss, test_acc, test_details = self._run_epoch(test_data,
                                                                [self._loss_op,
                                                                 self._correct_op,
                                                                 self._details_op])

        # Output test summary

        lu.info("Test Summary:")
        lu.log_value("Test", "loss: {:0.3f} accuracy: {:0.3f}%",
                     test_loss,
                     test_acc * 100)
        lu.log_value("Class 0 accuracy", "{}/{} ({:0.3f}%)",
                     test_details[0],
                     test_details[2],
                     test_details[0] / test_details[2] * 100.0)
        lu.log_value("Class 1 accuracy", "{}/{} ({:0.3f}%)",
                     test_details[1],
                     test_details[3],
                     test_details[1] / test_details[3] * 100.0 \
                     if test_details[3] != 0 else 0.0)
        lu.log_value("Test time", "{}", test_timer())

        summary.test_summary = SimpleNamespace(test_loss=float(test_loss),
                                               test_acc=float(test_acc),
                                               test_time=str(test_timer()))

        return summary

    # **************************************************

    def _run_batch(self,
                   x, xlen, y, ops,
                   learning_rate=None):
        """Run the model through one batch of data
        """

        # Build the input feeds

        feeds = {self._x: x,
                 self._xlen: xlen}

        if y is not None:
            feeds[self._y] = y

        if learning_rate is not None:
            feeds[self._learning_rate] = learning_rate

        # Add dropout if requested

        if self._dropout_prob is not None:
            if hasattr(self, '_training_op') and self._training_op in ops:
                feeds[self._dropout_prob] = self._config.parms.dropout_keep_probability
            else:
                feeds[self._dropout_prob] = 1.0

        # Setup summary and trace if requested

        run_options = None
        run_metadata = None

        if hasattr(self, '_summary_op'):

            if self._config.tf_trace and self._batch_no % SUMMARY_INTERVAL == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            ops.append(self._summary_op)

        # Run the operations and retrieve the outputs

        vals = self._session.run(ops,
                                 feeds,
                                 options=run_options,
                                 run_metadata=run_metadata)
        val_list = list(vals)

        # Save summary information

        if hasattr(self, '_summary_op'):

            summaries = val_list.pop()

            if self._batch_no % SUMMARY_INTERVAL == 0:
                if self._config.tf_trace:
                    self._summary_writer.add_run_metadata(run_metadata, 'Batch_{}'.format(self._batch_no))
                self._summary_writer.add_summary(summaries, self._batch_no)

            self._batch_no += 1

        return val_list

    # **************************************************

    def _run_epoch(self, data, ops,
                   random=False,
                   learning_rate=None):
        """Run the specified operations on the model across the entire dataset

        Intended for use with labeled data.
        """

        total_loss = 0.0
        total_correct = 0
        total_details = np.array([0] * 4)

        for i, (x, xlen, y, keys) in enumerate(cf.get_feature_batch(data,
                                                                    self._config.batch_size,
                                                                    self._features.n_features,
                                                                    self._features.packetlimit,
                                                                    random=random)):

            vals = self._run_batch(x, xlen, y, ops, learning_rate)

            for (op, val) in zip(ops, vals):
                if op is self._loss_op:
                    total_loss += val
                elif op is self._correct_op:
                    total_correct += val
                elif op is self._details_op:
                    total_details += np.array(val)

        return total_loss / (i + 1), \
            total_correct / ((i + 1) * self._config.batch_size), \
            total_details


# **************************************************

def _add_lstm_summaries(n_lstm_layers):
    """Add tensorboard summaries for the weights and bias values
    """

    for layer in range(n_lstm_layers):
        with tf.variable_scope("RNN/MultiRNNCell/Cell{}/BasicLSTMCell/Linear".format(layer),
                               reuse=True):
            _var_summaries("Matrix {}".format(layer), tf.get_variable("Matrix"))
            _var_summaries("Bias {}".format(layer), tf.get_variable("Bias"))

# **************************************************

def _var_summaries(title, var):
    """Add tensorboard summaries for the variable specified
    """

    def mk_name(vtype):
        return "{}/{}/{}".format(title, vtype, name).translate(str.maketrans(': ', '__'))

    name = var.name

    with tf.name_scope('summaries'):

        with tf.name_scope('mean'):
            mean = tf.reduce_mean(var)
        tf.summary.scalar(mk_name('Mean'), mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(mk_name('Stddev'), stddev)

        tf.summary.scalar(mk_name("Max"), tf.reduce_max(var))
        tf.summary.scalar(mk_name("Min"), tf.reduce_min(var))

        tf.summary.histogram(mk_name("Hist"), var)


# **************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
