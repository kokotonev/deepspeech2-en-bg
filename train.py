# Author: Nikola Tonev
# Date: March 2019

import os
import time
import utils
import shutil
import numpy as np 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.WARN)

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
    # if os.path.exists(hparams.log_dir):
    #     shutil.rmtree(hparams.log_dir)

    # Creating a new logging directory
    if not os.path.exists(hparams.log_dir):
        os.makedirs(hparams.log_dir)


    # Creating the graph structures for training and evaluation
    train_graph = tf.Graph()
    eval_graph = tf.Graph()



    ###################################
    ###  LOADING THE TRAINING DATA  ###
    ###################################

    # print('Loading data...\n')
    print('General preparation...\n\n')

    # Loading the training and testing data (including audio and transcriptions)
    # Transcriptions loaded as arrays of strings ---> e.g. ['0', '1', '2'] for 'abc'
    # train_audio, train_labels, test_audio, test_labels = utils.load_data(hparams.dataset, max_data=hparams.max_data)

    train_files, test_files = utils.load_filenames(hparams.dataset)
    # train_files = train_files[:10]
    # test_files = test_files[:10]
    print(' -> Filenames --- LOADED SUCCESSFULLY')

    # Loading the output mapping (in the form of a dictionary ---> e.g. {'a': 0, 'b': 1, 'c': 2})
    output_mapping = utils.load_output_mapping(hparams.dataset)
    print(' -> Output mapping --- LOADED SUCCESSFULLY')

    # Defining the output classes -> all characters from the output mapping + the blank character
    hparams.n_classes = len(output_mapping) + 1
    print(' -> Output classes --- LOADED SUCCESSFULLY')

    # Defining the longest of the training examples
    # hparams.input_max_len = max([max([len(x) for x in train_audio]), max([len(x) for x in test_audio])])

    # hparams.input_max_len, _ = utils.calculate_input_max_len(hparams.dataset)
    # hparams.input_max_len = 1081 # ---> for LibriSpeech (2597 for full set)
    # hparams.input_max_len = 1619 # ---> for BulPhonC
    hparams.input_max_len = 99 # ---> for Speech Commands
    print(' -> Maximum input length --- LOADED SUCCESSFULLY')


    # Padding the sequences so they are all with equal length --> new shape (max_data, max_length, n_features) 
    # train_audio = np.asarray(utils.pad_sequences(train_audio, hparams.input_max_len))
    # test_audio = np.asarray(utils.pad_sequences(test_audio, hparams.input_max_len))

    # Defining the number of frequency bins
    # hparams.n_features = train_audio.shape[2]

    if hparams.dataset == 'librispeech':
        arr = np.load('data/librispeech_processed/train-clean-100/19-198-0000.npy')
        hparams.n_features = arr[0].shape[1]
    elif hparams.dataset == 'bulphonc':
        arr = np.load('data/bulphonc_processed/audio/{}'.format(train_files[0]))
        hparams.n_features = arr.shape[1]
    elif hparams.dataset == 'speech_commands':
        arr = np.load('data/speech_commands_processed/train/backward-1.npy')
        hparams.n_features = arr.shape[1]

    print(' -> Number of features --- LOADED SUCCESSFULLY')

    # print('     +++ DATA LOADED +++')



    ##################################
    ###   INITIALIZING THE MODEL   ###
    ##################################

    print('\nInitializing model...\n')

    # Specifying the processor on which the model will be trained
    device = '/cpu:0'

    # Pulling some hyper-parameters from the config file
    epochs = hparams.n_epochs
    batch_size = hparams.batch_size
    steps_per_checkpoint = hparams.steps_per_checkpoint
    checkpoints_dirpath = hparams.checkpoints_dirpath

    if not os.path.exists(checkpoints_dirpath):
        os.makedirs(checkpoints_dirpath)

    # Creating training and evaluation sessions
    train_sess = tf.Session(graph=train_graph)
    # train_sess = tf_debug.TensorBoardDebugWrapperSession(train_sess, "Kokovich-2.local:6064")

    eval_sess = tf.Session(graph=eval_graph)
    # eval_sess = tf_debug.TensorBoardDebugWrapperSession(eval_sess, "Kokovich-2.local:6064")

    checkpoints_path = os.path.join(checkpoints_dirpath, 'checkpoint')

    # Creating a training model object
    with train_graph.as_default():
        with tf.device(device):
            if hparams.load_from_checkpoint == True:
                train_model = Model.load('model/hparams', 'model/checkpoints/checkpoint-846400', train_sess, model_modes.TRAIN)
            else:
                train_model = Model(hparams, model_modes.TRAIN)
                variables_initializer = tf.global_variables_initializer()


    # Creating an evaluation model object
    with eval_graph.as_default():
        with tf.device(device):
            if hparams.load_from_checkpoint == True:
                eval_model = Model.load('model/hparams', 'model/checkpoints/checkpoint-846400', eval_sess, model_modes.EVAL)
            else:
                eval_model = Model(hparams, model_modes.EVAL)

    # Creating summary logging file writers ---> will be populated later with methods like .add_summary()
    training_logger = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'train'), graph=train_graph)
    eval_logger = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'eval'), graph=eval_graph)


    ### Not sure if needed... if needed, pass as 'config' argument to tf.Session() ###
    # config = tf.ConfigProto()
    # config.log_device_placement = hparams.log_device_placement
    # config.allow_soft_placement = hparams.allow_soft_placement


    # Initializing all variables in the model
    if hparams.load_from_checkpoint == False:
        train_sess.run(variables_initializer)

    # Setting the global step and getting current time when training starts
    global_step = 0
    start_time = time.time()

    # Saving the initial state of the model ---> saving all hyper-parameters and variables from the model
    train_model.save(checkpoints_path, train_sess, global_step=0)


    print('\n\n     +++ MODEL INITIALIZED +++')




    ##################################
    ####    TRAINING EXECUTION    ####
    ##################################

    print('Training...')

    # Run until interrupted by the keyboard (user)
    try:

      #while epochs:
      for ep in range(epochs):

        # # The number of the current epoch = total epoch from config file - epochs left to run
        # current_epoch = hparams.n_epochs - epochs
        current_epoch = ep

        # Dividng the training data into batches and looping through them
        for i in range(int(len(train_files)/batch_size)):
            print('--{}'.format(i))
            curr_time = time.time()

            batch_train_audio = []
            batch_train_labels = []
            filenames = []

            # GET AUDIO FROM FILENAMES
            if hparams.dataset == 'librispeech':
                for file in train_files[i*batch_size:(i+1)*batch_size]:
                    arr = np.load('data/librispeech_processed/train-clean-100/{}'.format(file))
                    filenames.append(file)
                    batch_train_audio.append(arr[0])
                    batch_train_labels.append(arr[1])

            elif hparams.dataset == 'bulphonc':
                for file in train_files[i*batch_size:(i+1)*batch_size]:
                    audio = np.load('data/bulphonc_processed/audio/{}'.format(file))
                    label_id = file.split('-')[1][:-4]
                    label = np.load('data/bulphonc_processed/transcriptions/{}.npy'.format(label_id))
                    filenames.append(file)
                    batch_train_audio.append(audio)
                    batch_train_labels.append(label) 

            elif hparams.dataset == 'speech_commands':
                for file in train_files[i*batch_size:(i+1)*batch_size]:
                    audio = np.load('data/speech_commands_processed_reduced/train/{}'.format(file))
                    label_name = file.split('-')[0]
                    label = np.load('data/speech_commands_processed_reduced/transcriptions/{}.npy'.format(label_name))
                    filenames.append(file)
                    batch_train_audio.append(audio)
                    batch_train_labels.append(label)

            # PAD SEQUENCES
            batch_train_audio = np.asarray(utils.pad_sequences(batch_train_audio, hparams.input_max_len), dtype=np.float32)
            batch_train_labels = utils.sparse_tuple_from(np.asarray(batch_train_labels))

            # batch_train_audio = np.asarray(train_audio[i*batch_size:(i+1)*batch_size], dtype=np.float32)
            # batch_train_labels = utils.sparse_tuple_from(np.asarray(train_labels[i*batch_size:(i+1)*batch_size]))

            # Returns the cost value and the summary
            cost, _, summary = train_model.train(batch_train_audio, batch_train_labels, train_sess)

            # Updating the global step
            global_step += batch_size

            # Adding summary to the training logs
            training_logger.add_summary(summary, global_step=global_step)


            tot = time.time() - start_time
            h = int(tot/3600)
            m = int((tot/3600-h)*60)
            s = int((((tot/3600-h)*60)-m)*60)
            if h < 10: h = '0{}'.format(h)
            if m < 10: m = '0{}'.format(m)
            if s < 10: s = '0{}'.format(s)
            time_tot = '{}:{}:{}'.format(h, m, s)
            print('~~~ \nEpoch: {} \nGlobal Step: {} \nCost: {} \nTime: {} s\nTime total: {} \nFilenames: {}\n~~~'.format(current_epoch, global_step, cost, time.time() - curr_time, time_tot, filenames))


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

            # tf.get_variable_scope().reuse_variables()

        #if epochs > 0: epochs -= 1

    except KeyboardInterrupt:

        print('===========\nTotal iterations: {}\nAudio files processed: {}\nTime taken: {}\n==========='.format(i, global_step, time_tot))
        train_sess.close()
        eval_sess.close()

    print('===========\nTotal iterations: {}\nAudio files processed: {}\nTime taken: {}\n==========='.format(i, global_step, time_tot))
    train_sess.close()
    eval_sess.close()
