# Author: Nikola Tonev
# Date: March 2019

import os
import time
import utils
import argparse
import numpy as np
import soundfile as sf
import tensorflow as tf 


from config import hparams
from model import Model, model_modes


def infer():

    # Creating a command line argument parser
    arg_p = argparse.ArgumentParser()

    # Adding an expected argument
    arg_p.add_argument('-af', '--audiofile', required=True, type=str)

    # Parsing the arguments passed by the user
    args = arg_p.parse_args()

    # Creating an inference session
    infer_sess = tf.Session()

    # Creating an inference model
    model = Model.load('model/hparams', 'model/checkpoints/checkpoint-20', infer_sess, model_modes.INFER)

    # Loading the output mapping (in the form of a dictionary ---> e.g. {'a': 0, 'b': 1, 'c': 2})
    output_mapping = utils.load_output_mapping(hparams.dataset)

    # Reading the audio file from the path passed as a command line argument
    audio, sr = sf.read(args.audiofile)

    # Converting the audio to a spectrogram (feature representation)
    features = utils.compute_log_linear_spectrogram(audio, sr, window_size=20, step_size=10)

    # Padding the sequence so that it is the same length as all sequences on which the model was trained
    # Passing features in [] in order to make it three-dimensional (2D --> 3D)
    features_padded = utils.pad_sequences([features], model.config.input_max_len)


    # Setting the start time for decoding
    start_time = time.time()

    # Performing decoding ---> returns a ...
    decoded = model.infer(features_padded, infer_sess)

    # print(decoded)
    # print('###')
    # print(decoded[0][1])
    # print(output_mapping)
    # print('---end')


    # Converting transcription IDs ---> Text
    text_prediction = utils.ids_to_text(decoded[0][1], output_mapping)


    print('\n\n###############\nTranscription: {}\nTook {} seconds.'.format(text_prediction, time.time()-start_time))



if __name__ == '__main__':

    infer()