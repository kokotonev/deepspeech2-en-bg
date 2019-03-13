# Author: Nikola Tonev
# Date: March 2019

import os
import time
import utils
import shutil
import numpy as np 
import tensorflow as tf 

from model import Model, model_modes
from config import hparams

from tensorboard import default as tb_default
from tensorboard import program as tb_program


from tensorflow.python import debug as tf_debug


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
    #       TO DO !!!
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

    # Pulling some hyper-parameters from the config file
    epochs = hparams.n_epochs
    batch_size = hparams.batch_size
    steps_per_checkpoint = hparams.steps_per_checkpoint
    checkpoints_dirpath = hparams.checkpoints_dirpath

    if not os.path.exists(checkpoints_dirpath):
        os.makedirs(checkpoints_dirpath)

    checkpoints_path = os.path.join(checkpoints_dirpath, 'checkpoint')

    # Creating a training model object
    with train_graph.as_default():
        with tf.device(device):
            train_model = Model(hparams, model_modes.TRAIN)
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
    train_sess = tf_debug.TensorBoardDebugWrapperSession(train_sess, "Kokovich-2.local:6064")

    eval_sess = tf.Session(graph=eval_graph)
    eval_sess = tf_debug.TensorBoardDebugWrapperSession(eval_sess, "Kokovich-2.local:6064")

    # Initializing all variables in the model
    train_sess.run(variables_initializer)

    # Setting the global step and getting current time when training starts
    global_step = 0
    start_time = time.time()

    # Saving the initial state of the model ---> saving all hyper-parameters and variables from the model
    train_model.save(checkpoints_path, train_sess, global_step=0)







    ##################################
    ####    TRAINING EXECUTION    ####
    ##################################

    print('Training...')

    # Run until interrupted by the keyboard (user)
    try:

      while epochs:

        # The number of the current epoch = total epoch from config file - epochs left to run
        current_epoch = hparams.n_epochs - epochs

        # Dividng the training data into batches and looping through them
        for i in range(int(len(train_audio)/batch_size)):
            print('--{}'.format(i))

            batch_train_audio = np.asarray(train_audio[i*batch_size:(i+1)*batch_size], dtype=np.float32)
            batch_train_labels = utils.sparse_tuple_from(np.asarray(train_labels[i*batch_size:(i+1)*batch_size]))

            # Returns the cost value and the summary
            cost, _, summary = train_model.train(batch_train_audio, batch_train_labels, train_sess)

            # Updating the global step
            global_step += batch_size

            # Adding summary to the training logs
            training_logger.add_summary(summary, global_step=global_step)

            print('~~~ \nEpoch: {} \nGlobal Step: {} \nCost: {} \nTime: {} \n~~~'.format(current_epoch, global_step, cost, time.time() - start_time))


            # If the global step is a multiple of steps_per_checkpoint ---> if it is time for a checkpoint
            if global_step % steps_per_checkpoint == 0:

                print('Checkpointing... (Global step = {})'.format(global_step))

                # Saving a checkpoint after a certain number of iterations
                current_checkpoint = train_model.saver.save(train_sess, checkpoints_path, global_step=global_step)
                    
                # Immediately restorint the saved model to evaluate it!
                eval_model.saver.restore(eval_sess, current_checkpoint)

                # EVALUATING THE MODEL AT CHECKPOINT
                ler, summary = eval_model.eval(batch_train_audio, batch_train_labels, eval_sess)

                # Adding summary to the evaluation logs
                eval_logger.add_summary(summary, global_step=global_step)

                tf.summary.merge_all()

                print('#####\nEvaluation --- LER: {} %\n#####'.format(ler*100))

        if epochs > 0: epochs -= 1

    except KeyboardInterrupt:

      train_sess.close()
      eval_sess.close()


    train_sess.close()
    eval_sess.close()
