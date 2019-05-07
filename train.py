# Training execution script
#
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

    # Creating a new logging directory
    if not os.path.exists(hparams.log_dir):
        os.makedirs(hparams.log_dir)

    # Creating the graph structures for training and evaluation
    train_graph = tf.Graph()
    eval_graph = tf.Graph()



    ###################################
    ###  LOADING THE TRAINING DATA  ###
    ###################################

    print('General preparation...\n\n')

    # Loading the filenames of all training and testing files. Will be used for accessing the files later.
    train_files, test_files = utils.load_filenames(hparams.dataset)
    # train_files = ['filename'] # In case of single file overfitting, uncomment this line and replace 'filename' with the name of the corresponding file.
    print(' -> Filenames --- LOADED SUCCESSFULLY')

    # Loading the output mapping (in the form of a dictionary ---> e.g. {'a': 0, 'b': 1, 'c': 2})
    output_mapping = utils.load_output_mapping(hparams.dataset)
    print(' -> Output mapping --- LOADED SUCCESSFULLY')

    # Defining the output classes -> all characters from the output mapping + the blank character
    hparams.n_classes = len(output_mapping) + 1
    print(' -> Output classes --- LOADED SUCCESSFULLY')

    # Obtaining the length of the longest training example
    hparams.input_max_len, _ = utils.calculate_input_max_len(hparams.dataset)

    # Can be entered manually if only a part of a dataset is inteded to be used or for single file overfitting
    # hparams.input_max_len = 1081 # ---> for LibriSpeech (2597 for full set)
    # hparams.input_max_len = 1619 # ---> for BulPhonC
    # hparams.input_max_len = 99 # ---> for Speech Commands
    print(' -> Maximum input length --- LOADED SUCCESSFULLY')
 

    # Defining the number of frequency bins
    # Obtained by reading the shape of a random file from the dataset, since all of them are preporcessed in the same way and have the same frequency resolution.
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
    checkpoint_num = hparams.checkpoint_num

    # If the checkpoint path doesn't exist - creating it.
    if not os.path.exists(checkpoints_dirpath):
        os.makedirs(checkpoints_dirpath)

    # Creating training and evaluation sessions
    train_sess = tf.Session(graph=train_graph)

    eval_sess = tf.Session(graph=eval_graph)

    checkpoints_path = os.path.join(checkpoints_dirpath, 'checkpoint')


    # Creating a training model object
    with train_graph.as_default():
        with tf.device(device):
            # If training is resumed from a checkpoint
            if hparams.load_from_checkpoint == True:
                train_model = Model.load('model/hparams', 'model/checkpoints/checkpoint-{}'.format(checkpoint_num), train_sess, model_modes.TRAIN)
            
            # If training starts from scratch
            else:
                train_model = Model(hparams, model_modes.TRAIN)
                variables_initializer = tf.global_variables_initializer()


    # Creating an evaluation model object
    with eval_graph.as_default():
        with tf.device(device):
            # If training is resumed from a checkpoint
            if hparams.load_from_checkpoint == True:
                eval_model = Model.load('model/hparams', 'model/checkpoints/checkpoint-{}'.format(checkpoint_num), eval_sess, model_modes.EVAL)
            
            # If training starts from scratch
            else:
                eval_model = Model(hparams, model_modes.EVAL)

    # Creating summary logging file writers ---> will be populated later with methods like .add_summary()
    training_logger = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'train'), graph=train_graph)
    eval_logger = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'eval'), graph=eval_graph)


    # Initializing all variables in the model if training starts from scratch
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

      # Looping for as many times as epochs are specified
      for ep in range(epochs):

        # Defininf the number of the current epoch
        current_epoch = ep

        # Dividng the training data into batches and looping through them
        for i in range(int(len(train_files)/batch_size)):
            print('--{}'.format(i))
            curr_time = time.time()

            # Defining placeholders
            batch_train_audio = []
            batch_train_labels = []
            filenames = []

            # GET AUDIO FROM FILENAMES
            if hparams.dataset == 'librispeech':
                # Looping through the filenames for the current batch
                for file in train_files[i*batch_size:(i+1)*batch_size]:
                    # Reading in the .npy file for the training example
                    arr = np.load('data/librispeech_processed/train-clean-100/{}'.format(file))
                    filenames.append(file)
                    # Appending the audio and transcription to the batch arrays
                    batch_train_audio.append(arr[0])
                    batch_train_labels.append(arr[1])

            elif hparams.dataset == 'bulphonc':
                # Looping through the filenames for the current batch
                for file in train_files[i*batch_size:(i+1)*batch_size]:
                    # Loading the audio .npy file for the training example
                    audio = np.load('data/bulphonc_processed/audio/{}'.format(file))
                    label_id = file.split('-')[1][:-4]
                    # Loading the transcription .npy file for the training example
                    label = np.load('data/bulphonc_processed/transcriptions/{}.npy'.format(label_id))
                    filenames.append(file)
                    # Appending the audio and transcription to the batch arrays
                    batch_train_audio.append(audio)
                    batch_train_labels.append(label) 

            elif hparams.dataset == 'speech_commands':
                # Looping through the filenames for the current batch
                for file in train_files[i*batch_size:(i+1)*batch_size]:
                    # Loading the audio .npy file for the training example
                    audio = np.load('data/speech_commands_processed_reduced/train/{}'.format(file))
                    label_name = file.split('-')[0]
                    # Loading the transcription .npy file for the training example
                    label = np.load('data/speech_commands_processed_reduced/transcriptions/{}.npy'.format(label_name))
                    filenames.append(file)
                    # Appending the audio and transcription to the batch arrays
                    batch_train_audio.append(audio)
                    batch_train_labels.append(label)

            # Padding sequences so they are all with equal length --> new shape (max_data, max_lengt, n_features)
            batch_train_audio = np.asarray(utils.pad_sequences(batch_train_audio, hparams.input_max_len), dtype=np.float32)
            batch_train_labels = utils.sparse_tuple_from(np.asarray(batch_train_labels))


            # Run the training method from the model class. Returns the cost value and the summary.
            cost, _, summary = train_model.train(batch_train_audio, batch_train_labels, train_sess)

            # Updating the global step
            global_step += batch_size

            # Adding summary to the training logs
            training_logger.add_summary(summary, global_step=global_step)

            # Calculating time for the console output
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
                    
                # Immediately restoring the saved model to evaluate it!
                eval_model.saver.restore(eval_sess, current_checkpoint)

                # EVALUATING THE MODEL AT CHECKPOINT
                ler, summary = eval_model.eval(batch_train_audio, batch_train_labels, eval_sess)

                # Adding summary to the evaluation logs
                eval_logger.add_summary(summary, global_step=global_step)

                # Mergin all summaries
                tf.summary.merge_all()

                print('#####\nEvaluation --- LER: {} %\n#####'.format(ler*100))

    except KeyboardInterrupt:

        print('===========\nTotal iterations: {}\nAudio files processed: {}\nTime taken: {}\n==========='.format(i, global_step, time_tot))
        train_sess.close()
        eval_sess.close()

    print('===========\nTotal iterations: {}\nAudio files processed: {}\nTime taken: {}\n==========='.format(i, global_step, time_tot))
    train_sess.close()
    eval_sess.close()
