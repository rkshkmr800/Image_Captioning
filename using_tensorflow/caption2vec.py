'''

Program to create processed image data from pretrained VGG16

'''
import os
import pickle 
import numpy as np
import tensorflow as tf

#retrieving all the capitons related to images to dictionary, word2vec dictionary and max_length

max_length = 0
word2vec = {}
train_images_with_caption = {}
vocabulary_size = 0


pickle_in = open("processed_data/train_images_with_caption.pickle","rb")
train_images_with_caption = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("processed_data/word2vec.pickle","rb")
word2vec = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("processed_data/max_length.pickle","rb")
max_length = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("processed_data/vocabulary_size.pickle","rb")
vocabulary_size = pickle.load(pickle_in)
pickle_in.close()

#end of processing

padded_sequences = {}
subsequent_words = {}

print "processing captions"

for key in train_images_with_caption:

	caption = train_images_with_caption[key][0]
	caption_tokens = caption.split()

	caption_tokens_index = []
	partial_seqs = []
	next_words = []

	for i in caption_tokens:

		if i in word2vec.keys():
			caption_tokens_index.append(word2vec[i])
		else:
			caption_tokens_index.append(0)

	for i in range(1, len(caption_tokens_index)):		
		partial_seqs.append(caption_tokens_index[:i])
		next_words.append(caption_tokens_index[i])

	padded_partial_seqs = tf.keras.preprocessing.sequence.pad_sequences(partial_seqs, max_length, padding='post')
	#next_words_1hot = np.zeros([len(next_words), vocabulary_size], dtype=np.bool)

	#Vectorization
	#for i,next_word in enumerate(next_words):
	#	next_words_1hot[i, next_word] = 1

	padded_sequences[key] = padded_partial_seqs
	subsequent_words[key] = next_words

print "processing captions complete"   

pickle_out = open("processed_data/train_captions_vector.pickle","wb")
pickle.dump(padded_sequences, pickle_out)
pickle_out.close()

pickle_out = open("processed_data/train_nextword_vector.pickle","wb")
pickle.dump(subsequent_words, pickle_out)
pickle_out.close()