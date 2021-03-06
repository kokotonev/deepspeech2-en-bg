# Utility functions shared by multiple other files
#
# Author: Nikola Tonev
# Date: March 2019

import os
from scipy import signal
from config import hparams
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.WARN)


# Converting a mapped array of numbers back to text using the provided output mapping.
def ids_to_text(sequence, mapping):

	words = []
	# Looping through each character in the sentence
	for char in sequence:
		# Mapping the character to the keys from the output mapping
		for k, v in mapping.items():
			if char == v:
				words.append(k)
				break
	return ''.join(words)


# Computing the actual sequence lengths without the padding for all examples in a batch
def compute_seq_lens(input):

	# Keep only the highest frequency measurement for each time step --> flattening the 3rd dimension; convert to 1s and 0s
	used = tf.sign(tf.reduce_max(tf.abs(input), 2))

	# Sum all time steps where there is some frequency info (value is 1, not 0) --> flattening the 2nd dimension and giving an array of sequence lengths for all examples in the batch.
	lengths = tf.reduce_sum(used, 1)
	lengths = tf.cast(lengths, tf.int32)

	return lengths


# Padding all sequences in a batch, so they are all equal length (equal to the longest one)
def pad_sequences(sequences, max_len):

	padded_sequences = []

	for seq in sequences:
		# Creating zero-filled vectors of length n_features to make all sequences equal in length
		padding = [np.zeros(len(seq[1])) for _ in range(max_len - len(seq))]

		# If padding is needed, concatenate it to the sequence and append it to the return array
		if len(padding) > 0:
			padded_sequences.append(np.concatenate((seq, padding), axis=0))
		else:
			padded_sequences.append(seq)

	# Will return a tensor of shape (batch_size, max_len, n_features)
	return padded_sequences 


# Creating a sparse tensor for the labels (returning the 3 dense tensors that make up the sparse one)
def sparse_tuple_from(sequences, dtype=np.int32):

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


# Loading the output mapping for the corresponding data set
def load_output_mapping(dataset):

	output_mapping = {}

	# Defining the path to the output mapping file
	if dataset == 'librispeech':
		om_path = 'data/librispeech/output_mapping_en.txt'
	elif dataset == 'bulphonc':
		om_path = 'data/BulPhonC-Version3/output_mapping_bg.txt'
	else:
		raise ValueError('The chosen dataset is not supported!')

	# Populate the output mapping dictionary
	with open(om_path, 'r') as f:
		for line in f.readlines():
			key = line.split(' -> ')[0]
			val = int(line.split(' -> ')[1].strip('\n'))
			output_mapping[key] = val


	return output_mapping


# Compute the spectrogram of the passed audio file with based on the passed arguments
def compute_log_linear_spectrogram(audio, sample_rate, window_size=10, step_size=10, max_freq=None, eps=1e-10):

	# Making sure the step size is no bigger than the window size.
	if step_size > window_size:
		raise ValueError('The step size should be less than or equal to the window size!')

	# Calculating the number of samples per window and per step
	samples_per_winseg = int(round(0.001*window_size*sample_rate))
	samples_per_step = int(round(0.001*step_size*sample_rate))

	### z-score normalization
	audio = audio - np.mean(audio)
	audio = audio / np.std(audio)

	# Performing FFT to obrain the spectrogram of the audio signal
	f, t, spec = signal.spectrogram(audio, fs=sample_rate, window='hann', nperseg=samples_per_winseg, noverlap=samples_per_step, detrend=False)

	# Taking the log of the specgram values and transposing it (I guess so it could be fed by lines representing window segment into the RNN?)
	feats = np.log(spec.T.astype(np.float32) + eps)

	# Z-score normalization
	feats = (feats - np.mean(feats))/np.std(feats)

	return feats


# Loading the names for all files in the passed data set
def load_filenames(dataset):

	train_files = []
	test_files = []

	if dataset == 'librispeech':
		# Looping through files in the training directory
		for file in os.listdir('data/librispeech_processed/train-clean-100'):
			if file not in ['.', '..', '.DS_Store']:
				# Appending each file to the train_files array
				train_files.append(file)

		# Looping through file in the testing directory
		for file in os.listdir('data/librispeech_processed/test-clean'):
			if file not in ['.', '..', '.DS_Store']:
				# Appending each file to the test_files array
				test_files.append(file)

		return train_files, test_files

	elif dataset == 'bulphonc':

		all_files = []

		# Looping through all audio files in the dataset
		for file in os.listdir('data/bulphonc_processed/audio'):
			if file not in ['.', '..', '.DS_Store']:
				# Appending each file to the all_files array
				all_files.append(file)

		# Splitting all_files into train_files and test_files with the 80%/20% splitting practice
		train_files = all_files[:int(0.8*len(all_files))]
		test_files = all_files[int(0.8*len(all_files)):]

		return train_files, test_files


# Calculate the length of the longest training/testing file from the passed data set
def calculate_input_max_len(dataset):

	max_len = 0
	filename = ''

	if dataset == 'librispeech':

		# Defining the paths to the train and tests sets for LibriSpeech
		train_path = os.path.join(hparams.librispeech_path, 'train-clean-100')
		test_path = os.path.join(hparams.librispeech_path, 'test-clean')

		# Looping through all training files
		for file in os.listdir(train_path):

			if file not in ['.','..','.DS_Store']:
				train_arr = np.load(os.path.join(train_path, file));
				train_audio_len = train_arr[0].shape[0]
				# If the current file is longer than the max_len, update max_len
				if train_audio_len > max_len:
					max_len = train_audio_len
					filename = file

		# Looping through all testing files
		for file in os.listdir(test_path):

			if file not in ['.','..','.DS_Store']:
				test_arr = np.load(os.path.join(test_path, file))
				test_audio_len = test_arr[0].shape[0]
				# If the current file is longer than the max_len, update max_len
				if test_audio_len > max_len:
					max_len = test_audio_len
					filename = file


	return max_len, filename