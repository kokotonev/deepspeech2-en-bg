# Author: Nikola Tonev
# Date: March 2019

import os
import utils
import pickle

from tensorflow.contrib.rnn import GRUCell, MultiRNNCell

import tensorflow as tf
import memory_saving_gradients 

# tf.logging.set_verbosity(tf.logging.WARN)

# Clearing the default graph
tf.reset_default_graph()
tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

class model_modes:
    
    TRAIN = 1
    EVAL = 2
    INFER = 3



class Model(object):

    ### CONSTRUCTOR ###
    def __init__(self, config, mode):

        self.config = config
        self.mode = mode

        # Define the batch size
        batch_size = self.config.batch_size

        # If the system is in inference (actual recognition), change the batch size to 1 so it can recognize immediately
        if self.mode == model_modes.INFER:
            batch_size = 1

        # Creating placeholders for the input data and the labels
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.config.n_features], name='inputs') # Unkown number of audio files, unknown max length, n_features freq bins
        self.labels = tf.sparse_placeholder(tf.int32, name='labels')

        with tf.variable_scope("Compute_Sequence_Lengths"):
            # Computing the sequence lengths for all examples in the batch.
            seq_lens = utils.compute_seq_lens(self.inputs)



        # Setting dictionaries to hold wieight and bias matrices for the convolutional layers
        self.conv_weights={}
        self.conv_biases={}


        ### CREATING CONVOLUTIONAL LAYERS ###
        with tf.variable_scope("CNN"):

            # Transforming the input 3D -> 4D in order to pass it to the conv2d function
            conv_input = tf.reshape(self.inputs, [batch_size, self.config.input_max_len, self.config.n_features, 1])

            layer_output = 1

            for i in range(self.config.cnn_layers):

                # Setting the kernel (weight matrix)
                self.conv_weights['W_conv{}'.format(i+1)] = tf.Variable(tf.random_normal([5, 5, layer_output, 32*(i+1)]), name='W_conv{}'.format(i+1))

                #Setting the bias matrix
                self.conv_biases['b_conv{}'.format(i+1)] = tf.Variable(tf.random_normal([32*(i+1)]), name='b_conv{}'.format(i+1))

                # Update the kernel depth
                layer_output = 32*(i+1)

                # Applying the convolution layer on the input
                conv_input = tf.nn.conv2d(conv_input, self.conv_weights['W_conv{}'.format(i+1)], strides=[1, 1, 1, 1], padding='SAME')
                conv_input = tf.layers.batch_normalization(conv_input)

            # Reshaping the output of the convolutional layers
            conv_output = tf.reshape(conv_input, [batch_size, self.config.input_max_len, -1])





        ### CREATING RNN ###
        with tf.variable_scope("RNN"):

            # Creating forward and backward rnn layers, consisting of multiple layers with GRU cells
            self.fw_rnn_cell = MultiRNNCell([GRUCell(self.config.rnn_size) for _ in range(self.config.rnn_layers)])
            self.bw_rnn_cell = MultiRNNCell([GRUCell(self.config.rnn_size) for _ in range(self.config.rnn_layers)])

            # Setting the initial state of the forward and backward rnn layers; a zero-filled tensor
            self.fw_rnn_state = self.fw_rnn_cell.zero_state(batch_size, dtype=tf.float32)
            self.bw_rnn_state = self.bw_rnn_cell.zero_state(batch_size, dtype=tf.float32)

            # Constructing a bi-directional rnn layer from the forward and backward layers
            rnn_outputs, state = tf.nn.bidirectional_dynamic_rnn(self.fw_rnn_cell, self.bw_rnn_cell, conv_output, seq_lens, self.fw_rnn_state, self.bw_rnn_state, dtype=tf.float32)
            #
            # rnn_outputs - a tuple (output_fw, output_bw), containing the forward and backward rnn output Tensors.
            #               Those tensors are of size [batch_size, max_time, cell_fw/bw.output_size]
            # state - a tuple (output_state_fw, output_state_bw) containing the forward and backward final states of BRNN


            # ??? Maybe merging the forward and backward layers ???
            outputs = tf.reshape(rnn_outputs, [-1, self.config.rnn_size])





        ### CREATING FULLY CONNECTED LAYER ###
        with tf.variable_scope("Fully_Connected"):

            # Creating weight and bias matrices for the fully-connected layer
            W_fc = tf.Variable(tf.truncated_normal([self.config.rnn_size, self.config.n_classes], stddev=0.1), name='W_fc')
            b_fc = tf.Variable(tf.constant(0.1, shape=[self.config.n_classes]), name='b_fc')

            # Logit == the output layer of the RNN before going through the softmax activation function!
            # Multiplying the RNN output tensor with the weights matrix and adding the bias.
            logit = tf.matmul(outputs, W_fc) + b_fc

            # Reshaping and transposing...
            logit = tf.transpose(tf.reshape(logit, [batch_size, -1, self.config.n_classes]), (1, 0, 2))



        ####################



        # If in TRAINING mode
        if self.mode == model_modes.TRAIN:

            # Feeding the fully-connected output layer (a.k.a. the logit) into a CTC function
            loss = tf.nn.ctc_loss(self.labels, logit, seq_lens)

            # Calculating the mean of all values in the loss tensor.
            self.cost = tf.reduce_mean(loss)

            # Creating a Summary protocol buffer to write to an event file.
            cost_summary = tf.summary.scalar('cost', self.cost)
            # Mergin summaries
            self.summary = tf.summary.merge([cost_summary])

            # Creating a gradient descent optimizer --> returns an op that updates all trainable_variables to minimize the loss
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate).minimize(self.cost)



        # If in EVALUATION or INFERENCE mode (a.k.a. decoding)
        if self.mode == model_modes.EVAL or self.mode == model_modes.INFER:

            # Performing beam search decoding on the input sequence (logit) and returning decoded outputs.
            if self.config.beam_width > 0:
                self.decoded, self.log_probs = tf.nn.ctc_beam_search_decoder(logit, seq_lens, beam_width=self.config.beam_width)
            else:
                # ctc_greedy_decoder is special case of ctc_beam_search_decoder with beam_width = 1
                self.decoded, self.neg_sum_logits = tf.nn.ctc_greedy_decoder(logit, seq_lens)

        # Calculating the mean label error rate and writing a summary protocol to the event file if in evaluation mode
        if self.mode == model_modes.EVAL:
            with tf.variable_scope("LER_evaluation"):
                self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))
                ler_summary = tf.summary.scalar('label error rate', self.ler)
                self.summary = tf.summary.merge([ler_summary])


        # Creating a saver object, which will be used for saving and resotring the variables inside the model.
        self.saver = tf.train.Saver(max_to_keep=None)



    ##############################
    ### Defining class methods ###
    ##############################


    # A static method for loading a saved model
    @classmethod
    def load(cls, hparams_path, checkpoint_path, sess, mode):

        # Loading the hyperparameters from the passed filepath (.../config.py)
        with open(hparams_path, 'rb') as f:
            hparams = pickle.load(f)

        # Creating an object of this class with the obtained hparams and the passed mode
        obj = cls(hparams, mode)

        # Restore a saved model from the passed filepath
        obj.saver.restore(sess, checkpoint_path)

        # with sess as ses:
        #     print('@@@@@@@@ ---> {}'.format(obj.conv_weights['W_conv1'].eval()))

        return obj


    # A method for training the model
    def train(self, inputs, targets, sess):

        # Make sure we are in training mode
        assert self.mode == model_modes.TRAIN

        tf.reset_default_graph()

        # Train the model with the passed inputs and target labels
        return sess.run([self.cost, self.optimizer, self.summary], feed_dict={self.inputs: inputs, self.labels: targets})


    # A method for evaluating the model
    def eval(self, inputs, targets, sess):

        # Make sure we are in evaluation mode
        assert self.mode == model_modes.EVAL

        # Evaluate the model with the passed inputs and target labels
        return sess.run([self.ler, self.summary], feed_dict={self.inputs: inputs, self.labels: targets})


    # A method for decoding an audio file (inference)
    def infer(self, inputs, sess):

        # Make sure we are in inference mode
        assert self.mode == model_modes.INFER

        # init = tf.global_variables_initializer()
        # sess.run(init)
        return sess.run([self.decoded], feed_dict={self.inputs: inputs})


    # A method for saving the model
    def save(self, path, sess, global_step=None):

        # Save the hyperparameters first
        with open(os.path.join('model/hparams'), 'wb') as f:
            pickle.dump(self.config, f)

        # Save the current state of the model
        self.saver.save(sess, path, global_step=global_step)

















