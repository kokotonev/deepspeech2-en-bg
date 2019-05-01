# Author: Nikola Tonev
# Date: April 2019


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import utils
import math
import os

def get_smallest_sd(sentences, dataset):

	stdevs = []
	l_sentences = [];

	for tup in sentences:
		idx = tup[0]
		sentence = tup[1]

		if (dataset == 'librispeech' and (len(sentence) > 160)) or (dataset == 'bulphonc' and (len(sentence) > 60)):
			l_sentences.append([idx, sentence])

			distribution = []

			# Creating an empty array to hold the distribution values later
			if dataset == 'librispeech':
				for i in range(27):
					distribution.append(0)
			elif dataset == 'bulphonc':
				for i in range(30):
					distribution.append(0)

			# Populating the distribution array by incrementing the according element when a letter is seen.
			for l in sentence:
				if l != ' ':
					distribution[output_mapping[l]] += 1

			mean = sum(distribution)/len(distribution)
			variance = 0
			for el in distribution:
				variance += (mean - el)**2
			variance = variance/len(distribution)

			stdev = math.sqrt(variance)

			stdevs.append(stdev)

	# print(stdevs)
	stdevs_sorted = stdevs.copy()
	stdevs_sorted.sort()
	flattest = stdevs_sorted[:10]
	idx_ = []
	for s in flattest:
		idx_.append(l_sentences[stdevs.index(s)][0])

	idx_largest = []
	for l in stdevs_sorted[-10:]:
		idx_largest.append(l_sentences[stdevs.index(l)][0])
	print(stdevs_sorted[-10:])
	print(idx_largest)
	print('----------')

	return flattest, idx_






def map_sentence(sentence, dataset):

	alphabet = []
	distribution = []

	if dataset == 'librispeech':
		nums = 27
	elif dataset == 'bulphonc':
		nums = 30

	for i in range(nums):
		alphabet.append(i)
		distribution.append(0)


	for j in sentence:
		if j != ' ':
			distribution[output_mapping[j]] += 1


	# print(distribution)
	# distribution = np.exp(distribution)/sum(np.exp(distribution))
	# print(distribution)



	plt.figure(1, figsize=(14, 8))
	plt.suptitle('Number of occurences of each letter in the sentence:')

	letsss = []
	for k, v in output_mapping.items():
		letsss.append(k)

	plt.plot(alphabet, distribution, 'ro-')
	plt.grid(True)
	plt.xticks(alphabet, letsss)
	plt.yticks(range(75))
	plt.ylabel('Number of occurences')

	# plt.axis([0,max(distribution),0,1])

	plt.show()



def calculate_avg_dist(dataset):

	sentences = []
	lengths = []
	letters = []
	s_letters = []
	alphabet = []
	distribution = []


	###################
	### LIBRISPEECH ###
	###################

	if dataset == 'librispeech':

		# Creating an array to hold the numbers corresponding to the output mapping
		for i in range(27):
			alphabet.append(i)

		# Walking through all folders in LibriSpeech
		for root, dirs, files in tqdm(list(os.walk('data/librispeech'))):

			if files and root != 'data/librispeech':
				# Going through all files in each folder
				for file in files:
					if file[-4:] == '.txt':
						with open(os.path.join(root, file)) as f:
							for line in f.readlines():
								stc_id = line.split(' ', 1)[0]
								sentence = line.split(' ', 1)[1].strip('\n').lower()
								sentences.append([stc_id, sentence])

		counter = 0
		sample = []

		flattest, flt_idx = get_smallest_sd(sentences, dataset)
		print(flattest)
		print(flt_idx)


		##############################################
		### Looping from all sentences in LibriSpeech

		for stc in sentences:

			if (len(stc[1]) == 160):
				counter += 1

			# Appending the lengths of all sentences to an array
			lengths.append(len(stc[1]))

			distribution = []

			# Creating an empty array to hold the distribution values later
			for i in range(27):
				distribution.append(0)

			# Populating the distribution array by incrementing the according element when a letter is seen.
			for ltr in stc[1]:
				if ltr != ' ':
					distribution[output_mapping[ltr]] += 1

			# Adding all distributions to an array.
			letters.append(distribution)


		avg_ltrs = []

		########################################
		### Looping through the alphabet lengths

		for i in range(len(distribution)):
			lets = [];
			for dist in letters:
				lets.append(dist[i]) # Appending the occurences of each same letter to an array.

			avg_ltrs.append(sum(lets)/len(lets)) # Calculating the average of those occurences


		avg_len = sum(lengths)/len(lengths) # Calculating the average length of sentences

		print('Average length of sentence: {} characters.\nNumber of sentences with that length: {}'.format(avg_len, counter))	


		####################################

		plt.figure(1, figsize=(14, 8))
		plt.suptitle('Average occurence of each letter in the whole dataset (Librispeech)')

		letsss = []
		for k, v in output_mapping.items():
			letsss.append(k)

		plt.subplot(121)
		plt.plot(avg_ltrs, 'ro-')
		plt.grid(True)
		plt.xticks(alphabet, letsss)
		plt.ylabel('Average number of occurences')


		plt.subplot(122)
		plt.bar(alphabet, avg_ltrs)
		plt.grid(True)
		plt.xticks(alphabet, letsss)
		plt.ylabel('Average number of occurences')

		plt.show()


	##################
	###  BULPHONC  ###
	##################

	elif dataset == 'bulphonc':

		# Creating an array to hold the numbers corresponding to the output mapping
		for i in range(30):
			alphabet.append(i)

		# Looping through all transcriptions encoding them and saving the in the relevant subdirectory.
		for filename in tqdm(os.listdir('data/BulPhonC-Version3/sentences')):
			if filename[-4:] == '.txt':
				with open('data/BulPhonC-Version3/sentences/{}'.format(filename)) as f:
					lines = f.readlines()
					stc_id = filename[:-4]
					sentence = lines[0].strip('\n')
					sentences.append([stc_id, sentence])


		counter = 0
		sample = []

		flattest, flt_idx = get_smallest_sd(sentences, dataset)
		print(flattest)
		print(flt_idx)


		###########################################
		### Looping from all sentences in BulPhonC

		for stc in sentences:

			if (len(stc[1]) == 61):
				counter += 1

			# Appending the lengths of all sentences to an array
			lengths.append(len(stc[1]))

			distribution = []

			# Creating an empty array to hold the distribution values later
			for i in range(30):
				distribution.append(0)

			# Populating the distribution array by incrementing the according element when a letter is seen.
			for ltr in stc[1]:
				if ltr != ' ':
					distribution[output_mapping[ltr]] += 1

			# Adding all distributions to an array.
			letters.append(distribution)


		avg_ltrs = []

		########################################
		### Looping through the alphabet lengths

		for i in range(len(distribution)):
			lets = [];
			for dist in letters:
				lets.append(dist[i]) # Appending the occurences of each same letter to an array.

			avg_ltrs.append(sum(lets)/len(lets)) # Calculating the average of those occurences


		avg_len = sum(lengths)/len(lengths) # Calculating the average length of sentences

		print('Average length of sentence: {} characters.\nNumber of sentences with that length: {}'.format(avg_len, counter))	


		####################################

		plt.figure(1, figsize=(14, 8))
		plt.suptitle('Average occurence of each letter in the whole dataset (BulPhonC)')

		letsss = []
		for k, v in output_mapping.items():
			letsss.append(k)

		plt.subplot(121)
		plt.plot(avg_ltrs, 'ro-')
		plt.grid(True)
		plt.xticks(alphabet, letsss)
		plt.ylabel('Average number of occurences')


		plt.subplot(122)
		plt.bar(alphabet, avg_ltrs)
		plt.grid(True)
		plt.xticks(alphabet, letsss)
		plt.ylabel('Average number of occurences')

		plt.show()







if __name__ == '__main__':

	output_mapping = utils.load_output_mapping('librispeech')

	calculate_avg_dist('librispeech')

	# arg_p = argparse.ArgumentParser()

	# arg_p.add_argument('-stc', '--sentence', required=True, type=str)

	# args = arg_p.parse_args()

	# sentence = args.sentence.lower()

	# map_sentence(sentence, 'bulphonc')
