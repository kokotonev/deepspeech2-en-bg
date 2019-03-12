# Author: Nikola Tonev
# Date: March 2019

import os
import utils
import shutil
import numpy as np 
import tensorflow as tf 

from model import Model, model_modes
from config import hparams

from tensorboard import default as tb_default
from tensorboard import program as tb_program

# Removes nodes from the default graph and reset it
tf.reset_default_graph()


if __name__ == '__main__':

	
	#############################
    ###  GENERAL PREPARATION  ###
    #############################


    # Checking for an existing logging directory and deleting it if it exists
    if os.path.exists(hparams.log_dir):
    	shutil.rmtree(hparams.log_dir)

    # Creating a new logging directory
    os.makedirs(hparams.log_dir)


    ### CONFIGURE TENSORBOARD ###
    #
    #
    #		TO DO !!!
    #
    #
    ############################


    # Creating the graph structures for training and evaluation
    train_graph = tf.Graph()
    eval_graph = tf.Graph()



    ###################################
    ###  LOADING THE TRAINING DATA  ###
    ###################################

    print('Loading data...')

    # Loading the training and testing data (including audio and transcriptions)
    # Transcriptions loaded as arrays of strings ---> e.g. ['0', '1', '2'] for 'abc'
    train_audio, train_labels, test_audio, test_labels = utils.load_data(hparams.dataset, max_data=hparams.max_data)

    # Loading the output mapping (in the form of a dictionary ---> e.g. {'a': 0, 'b': 1, 'c': 2})
    output_mapping = utils.load_output_mapping(hparams.dataset)

    # Defining the output classes -> all characters from the output mapping + the blank character
    hparams.n_classes = len(output_mapping) + 1

    # Defining the longest of the training examples
    hparams.input_max_len = max([max([len(x) for x in train_audio]), max([len(x) for x in test_audio])])

    # Padding the sequences so they are all with equal length --> new shape (max_data, max_length, n_features) 
    train_audio = np.asarray(utils.pad_sequences(train_audio, hparams.input_max_len))
    test_audio = np.asarray(utils.pad_sequences(test_audio, hparams.input_max_len))

    # Defining the number of frequency bins
    hparams.n_features = train_audio.shape[2]





    ##################################
    ###   INITIALIZING THE MODEL   ###
    ##################################

    print('Initializing model...')

    # Specifying the processor on which the model will be trained
    device = '/cpu:0'


    # Creating a training model object
    with train_graph.as_default():
    	with tf.device(device):
    		training_model = Model(hparams, model_modes.TRAIN)
    		variables_initializer = tf.global_variables_initializer()


    # Creating an evaluation model object
    with eval_graph.as_default():
    	with tf.device(device):
    		eval_model = Model(hparams, model_modes.EVAL)


    # Creating summary logging file writers ---> will be populated later with methods like .add_summary()
    training_logger = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'train'), graph=train_graph)
    eval_logger = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'eval'), graph=eval_graph)


    ### Not sure if needed... if needed, pass as 'config' argument to tf.Session() ###
    # config = tf.ConfigProto()
    # config.log_device_placement = hparams.log_device_placement
    # config.allow_soft_placement = hparams.allow_soft_placement


    # Creating training and evaluation sessions
    train_sess = tf.Session(graph=train_graph)
    eval_sess = tf.Session(graph=eval_sess)


    ##### TODO: 
    #
    # - Finish model initialization
    # - Write script for executing the training process





















