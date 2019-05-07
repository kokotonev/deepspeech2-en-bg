# Script for preprocessing the raw flac files and their corresponding transcriptions
# from LibriSpeech to '.npy' format files suitable for the model to read.
#
# Author: Nikola Tonev
# Date: March 2019

import soundfile as sf

import os
import numpy as np 
import argparse
import utils
from tqdm import tqdm



def preprocess_librispeech():

	# Loading the English alphabet output mapping
	output_mapping = utils.load_output_mapping('librispeech')

	# Creating a directory for the processed data if it doesn't exits yet.
	if not os.path.exists('data/librispeech_processed'):
		os.makedirs('data/librispeech_processed')

	# Creating subdirectories for the different parts of the data set if they don't exist yet.
	dirpaths = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100']
	for dirpath in dirpaths:
		if not os.path.exists('data/librispeech_processed/{}'.format(dirpath)):
			os.makedirs('data/librispeech_processed/{}'.format(dirpath))

	# Placeholders
	audio_feats = {}
	transcripts = {}

	print('\n ####  PREPROCESSING LIBRISPEECH CORPUS  ####\n')

	# Go through all files and folders under 'data/librispeech'
	for root, dirs, files in tqdm(list(os.walk('data/librispeech'))):
		# If files exist and the path is not to the master directory... 
		if files and root != 'data/librispeech':
			# Loop through all files in the directory
			for file in files:
				if file[-5:] == '.flac': # If it is an audio file...
					# Read the audio file
					audio, sr = sf.read(os.path.join(root, file))
					# Compute its spectrogram
					features = utils.compute_log_linear_spectrogram(audio, sr, window_size=20, step_size=10)
					# Append the spectrogram to the audio_feats dictionary
					audio_feats[file[:-5]] = features
				elif file[-4:] == '.txt': # If it is a text file (transcription)...
					with open(os.path.join(root, file)) as f:
						# Loop through all lines and slice them up to get the ids and transcriptions
						for line in f.readlines():
							audio_file_id = line.split(' ', 1)[0]
							transcription = line.split(' ', 1)[1].strip('\n').lower()
							transcription_mapped = []
							# Map the transcription according to the output mapping
							for l in transcription:
								transcription_mapped.append(str(output_mapping[l]))
							# Append the transcription to the transcripts dictionary
							transcripts[audio_file_id] = transcription_mapped

			# Setting the folder to which the current .npy files will be saved
			for dirpath in dirpaths:
				if root[17:17+len(dirpath)] == dirpath:
					fold = dirpath + '/'
					break

			# Saving all the info from the dictionaries above to the according folder
			for key in audio_feats.keys():
				save_path = 'data/librispeech_processed/' + fold + key + '.npy'
				if not os.path.exists(save_path):
					np.save(save_path, [audio_feats[key], transcripts[key]])

			# Clearing the dictionaries to prepare them for the next iteration of the loop
			audio_feats.clear()
			transcripts.clear()

	print('\n  --- PREPROCESSING COMPLETED ---\n')


if __name__ == '__main__':
	preprocess_librispeech()