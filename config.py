# Author: Nikola Tonev
# Date: 06 March 2019

from tensorflow.contrib.training import HParams

hparams = HParams(

	batch_size = 3,
	n_epochs = 3,
	steps_per_checkpoint = 10,
	checkpoints_dirpath = 'model/checkpoints',
	learning_rate = 0.001,
	log_dir='logs',	
	max_data = 0,


	# Data config
	dataset='librispeech',
	librispeech_path='data/librispeech_processed',
	bulphonc_path='data/bulphonc_processed',

	# CNN config
	cnn_layers = 2,

	# RNN config
	rnn_layers = 3,  # Best: 7 -- possible upgrade
	rnn_size = 128,

	# Decoder config
	beam_width = 32
)
