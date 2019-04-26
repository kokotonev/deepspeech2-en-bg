# Author: Nikola Tonev
# Date: April 2019

import soundfile as sf 

import os
import numpy as np 
import argparse
import utils
from tqdm import tqdm

def preprocess_speech_commands():
	
	# Loading the English alphabet output mapping
	output_mapping = utils.load_output_mapping('speech_commands')

	# Creating a directory for the processed data if it doesn't exist yet
	if not os.path.exists('data/speech_commands_processed_reduced'):
		os.makedirs('data/speech_commands_processed_reduced')


	# Creating subdirectories for the train and test sets
	dirpaths = ['train', 'test', 'transcriptions']
	for dirpath in dirpaths:
		if not os.path.exists('data/speech_commands_processed_reduced/{}'.format(dirpath)):
			os.makedirs('data/speech_commands_processed_reduced/{}'.format(dirpath))



	print('\n ### PREPROCESSING SPEECH COMMANDS DATASET ###\n')

	# Go through all files and folders under data/speech_commands
	for root, dirs, files in tqdm(list(os.walk('data/speech_commands'))):
		# If files exist and the path is not to the master directory...
		root_paths = ['data/speech_commands/go', 'data/speech_commands/stop', 'data/speech_commands/left', 'data/speech_commands/right', 'data/speech_commands/follow', 'data/speech_commands/up', 'data/speech_commands/down', 'data/speech_commands/forward', 'data/speech_commands/yes', 'data/speech_commands/no']
		# if files and root != 'data/speech_commands' and root != 'data/speech_commands/_background_noise_':
		if files and root in root_paths:

			# Getting the transcription from the folder name
			transcription = root.split('/')[2]
			transcription_mapped = []
			for l in transcription:
				transcription_mapped.append(str(output_mapping[l]))

			np.save('data/speech_commands_processed_reduced/transcriptions/{}.npy'.format(transcription), transcription_mapped)

			idx = 0

			# Looping through all files in each folder in order to get the audio recordings
			for file in files:
				if file[-4:] == '.wav':
					audio, sr = sf.read(os.path.join(root, file))
					features = utils.compute_log_linear_spectrogram(audio, sr, window_size=20, step_size=10)

				idx += 1

				# Splitting data into train and test set (80%/20%) and saving it
				if idx <= 0.8*len(files):
					np.save('data/speech_commands_processed_reduced/train/{}-{}.npy'.format(transcription, idx), features)
				else:
					np.save('data/speech_commands_processed_reduced/test/{}-{}.npy'.format(transcription, idx), features)


	print('\n  --- PREPROCESSING COMPLETED ---\n')


if __name__ == '__main__':
	preprocess_speech_commands()