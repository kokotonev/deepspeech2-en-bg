# Author: Nikola Tonev
# Date: March 2019

import os
from scipy import signal
from config import hparams
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.WARN)

def ids_to_text(sequence, mapping):

	words = []
	for char in sequence:
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

		if len(padding) > 0:
			padded_sequences.append(np.concatenate((seq, padding), axis=0))
		else:
			padded_sequences.append(seq)

	# Will return a tensor of shape (batch_size, max_len, n_features)
	return padded_sequences 


# Creating a sparse tensor for the labels (returnin the 3 dense tensors that make up the sparse one)
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



def load_output_mapping(dataset):

	output_mapping = {}

	if dataset == 'librispeech':
		om_path = 'data/librispeech/output_mapping_en.txt'
	elif dataset == 'bulphonc':
		om_path = 'data/BulPhonC-Version3/output_mapping_bg.txt'
	else:
		raise ValueError('The chosen dataset is not supported!')


	with open(om_path, 'r') as f:
		for line in f.readlines():
			key = line.split(' -> ')[0]
			val = int(line.split(' -> ')[1].strip('\n'))
			output_mapping[key] = val


	return output_mapping


def compute_log_linear_spectrogram(audio, sample_rate, window_size=10, step_size=10, max_freq=None, eps=1e-10):

	# # Making sure the maximum frequency is no larger than half of the sampling frequency.
	# if max_freq = None:
	# 	max_freq=sample_rate/2
	# if max_freq > sample_rate/2:
	# 	raise ValueError('The highest frequency must be less than half of the sample rate!')

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





def load_data(dataset, max_data=0, train_size=0.8):

	if dataset == 'librispeech':

		# Defining the paths to the train and tests sets for LibriSpeech
		train_path = os.path.join(hparams.librispeech_path, 'train-clean-100')
		test_path = os.path.join(hparams.librispeech_path, 'test-clean')

		# Defininf the max data for training and testing
		max_data_train = int(max_data*train_size)
		max_data_test = int(max_data*(1-train_size))

		train_data = []
		test_data = []

		# Loading training data
		for file in os.listdir(train_path):
			
			# Limiting the max number of files that will be loaded.
			if max_data > 0 and len(train_data) >= int(max_data_train):
				train_data = train_data[:max_data_train]
				break

			arr = np.load(os.path.join(train_path, file))
			audio = arr[0]
			label = arr[1]
			train_data.append((audio, label))
		

		# Loading test data
		for file in os.listdir(test_path):

			# Limiting the max number of files that will be loaded.
			if max_data > 0 and len(test_data) >= int(max_data_test):
				test_data = test_data[:max_data_test]
				break

			arr = np.load(os.path.join(test_path, file))
			audio = arr[0]
			label = arr[1]
			test_data.append((audio, label)) 


		train_audio = [x[0] for x in train_data]
		train_labels = [x[1] for x in train_data]
		test_audio = [x[0] for x in test_data]
		test_labels = [x[1] for x in test_data]

		return train_audio, train_labels, test_audio, test_labels



	elif dataset == 'bulphonc':

		# TO DO 
		pass


	else:
		raise ValueError("Invalid dataset name! Should be either 'librispeech' or 'bulphonc'! ")