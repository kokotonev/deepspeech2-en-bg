import os
from scipy import signal
import numpy as np


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