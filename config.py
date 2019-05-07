# Author: Nikola Tonev
# Date: March 2019

from tensorflow.contrib.training import HParams

hparams = HParams(

	batch_size = 32,
	n_epochs = 20,
	steps_per_checkpoint = 160,
	checkpoints_dirpath = 'model/checkpoints',
	learning_rate = 0.025,
	log_dir='logs',	
	max_data = 0,
	load_from_checkpoint = False,
	checkpoint_num = 88460,


	# Data config
	dataset='speech_commands', # 'librispeech'/'bulphonc'/'speech_commands'
	librispeech_path='data/librispeech_processed',
	bulphonc_path='data/bulphonc_processed',
	speech_commands_path = 'data/speech_commands_processed_reduced',

	# CNN config
	cnn_layers = 2,

	# RNN config
	rnn_layers = 3,  # Best: 7 -- possible upgrade
	rnn_size = 128,

	# Decoder config
	beam_width = 32
)
