# Author: Nikola Tonev
# Date: 3 March 2019


import os
import numpy as np 
import utils
from tqdm import tqdm


def preprocess_bulphonc():

	# Loading the Bulgarian alphabet output mapping
	output_mapping = utils.load_output_mapping('bulphonc')

	# Creating a directory for the processed data if it doesn't exist yet
	if not os.path.exists('data/bulphonc_processed'):
		os.makedirs('data/bulphonc_processed')

	# Creating subdirectories for the transcriptions and the audio files
	dirpaths = ['transcriptions', 'audio']
	for dirpath in dirpaths:
		if not os.path.exists('data/bulphonc_processed/{}'.format(dirpath)):
			os.makedirs('data/bulphonc_processed/{}'.format(dirpath))

	print('\n ####  PREPROCESSING BULPHONC CORPUS  ####\n')

	print('Transcriptions...')
	# Looping through all transcriptions encoding them and saving the in the relevant subdirectory.
	for filename in tqdm(os.listdir('data/BulPhonC-Version3/sentences')):
		if filename[-4:] == '.txt':
			with open('data/BulPhonC-Version3/sentences/{}'.format(filename)) as f:
				lines = f.readlines()
				transcription = lines[0].strip('\n')
				transcription_mapped = []
				for l in transcription:
					transcription_mapped.append(str(output_mapping[l]))
				np.save('data/bulphonc_processed/transcriptions/{}.npy'.format(filename[:-4]), transcription_mapped)

	print('\nAudio...')
	# Looping through all audio recordings of all speakers, encoding them and saving them in the relevant subdirectory
	for root, dirs, files in tqdm(list(os.walk('data/BulPhonC-Version3/speakers'))):
		if files:
			for file in files:
				if file[_4:] == '.wav':
					audio, sr = sf.read(os.path.join(root, file))
					features = utils.compute_log_linear_spectrogram(audio, sr, window_size=20, step_size=10)
					np.save('data/bulphonc_processed/audio/S{}-{}.npy'.format(root[39:], file[:-4]), features)

	print('\n  --- PREPROCESSING COMPLETED ---\n')

if __name__ == '__main__':
	preprocess_bulphonc()