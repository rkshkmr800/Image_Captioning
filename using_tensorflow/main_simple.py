import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import pickle
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import argparse

tf.set_random_seed(2)

num_units = 128
learning_rate = 0.001
epochs = 5
batch_size = 512

pickle_in = open("processed_data/vocabulary_size.pickle","rb")
vocabulary_size = pickle.load(pickle_in)
pickle_in.close() 

pickle_in = open("processed_data/train_images_processed.pickle","rb")
train_images_processed = pickle.load(pickle_in)
pickle_in.close() 

pickle_in = open("processed_data/train_captions_vector.pickle","rb")
train_captions_vector = pickle.load(pickle_in)
pickle_in.close() 

pickle_in = open("processed_data/train_nextword_vector.pickle","rb")
train_nextword_vector = pickle.load(pickle_in)
pickle_in.close() 

max_length = 64
embedding_size = 128

class NNModel(object):
	def __init__(self):

		###############
		# input data  #
		###############
		self.encoder_emb_inp = tf.placeholder(tf.float32,[64,None,64])
		self.decoder_emb_inp = tf.placeholder(tf.int32,[None,max_length])
		self.decoder_labels = tf.placeholder(tf.float32,[None,vocabulary_size])
		
		# Build RNN cell
		self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

		self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.encoder_cell, self.encoder_emb_inp, dtype=tf.float32, time_major=True)

		# Embedding
		self.embeddings = tf.get_variable("embeddings", [vocabulary_size, embedding_size])

		# 	Look up embedding:
		#   decoder_inputs: [max_time, batch_size]
		#   decoder_emb_inp: [max_time, batch_size, embedding_size]
		self.decoder_inp = tf.nn.embedding_lookup(self.embeddings, tf.transpose(self.decoder_emb_inp))
		
		self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

		self.init_state = tf.contrib.rnn.LSTMStateTuple(self.encoder_outputs[-1],self.encoder_outputs[-1])

		self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(self.decoder_cell, self.decoder_inp ,initial_state=self.init_state,dtype=tf.float32, time_major=True,scope="proj_decoder")

		self.logits = tf.contrib.layers.fully_connected(self.decoder_outputs[-1], vocabulary_size,activation_fn=None,weights_initializer=tf.initializers.random_uniform(-0.1,0.1),biases_initializer=tf.initializers.random_uniform(-0.1,0.1))

		self.prediction = tf.nn.softmax(self.logits)

		self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.decoder_labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.decoder_labels, logits=self.logits))

		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


def train():

	images = []
	captions = []
	next_words = []

	images_count = 0

	for i in train_images_processed.keys():
		for j in range(0,len(train_captions_vector[i])):
			images.append(train_images_processed[i])
			captions.append(train_captions_vector[i][j])
			next_words.append(train_nextword_vector[i][j])

		images_count += 1
		if images_count == 1000:
			break

	path = "weights/model_simple.ckpt"
	if os.path.isfile((path+".meta")):
		print "Loading the model..."
		saver = tf.train.Saver()
		savePath = saver.restore(sess, path)
		print "Loading complete"
	else:
		print "creating the model..."
		saver = tf.train.Saver()

	for k in range(0,epochs):

		start_time = time.time()

		data_set_size = len(images)
		sum_cost = 0
		sum_accuracy = 0
		batch_count = 0
		for j in xrange(0,data_set_size,batch_size):

			if (data_set_size-j) < batch_size:
				break

			img = images[j:j+batch_size]
			img = np.reshape(img,[-1,64,64])

			images_batch = []
			for i in range(0,len(img[0])):
				temp = []
				for j in range(0,len(img)):
					temp.append(img[j][i])
				images_batch.append(temp)

			next_words_batch = []
			for t in range(0,batch_size):
				next_words_1hot = [0] * vocabulary_size
				next_words_1hot[next_words[j+t]] = 1
				next_words_batch.append(next_words_1hot)

			captions_batch = []
			for t in range(0,batch_size):
				cap = []
				for i in captions[j]:
					cap.append(i)
				for i in range(0,24):
					cap.append(0)
				captions_batch.append(cap)


			_,cost,accuracy = sess.run([nn.optimizer,nn.cost,nn.accuracy],feed_dict={nn.encoder_emb_inp:images_batch,nn.decoder_emb_inp:captions_batch,nn.decoder_labels:next_words_batch})
			sum_cost += cost
			sum_accuracy += accuracy
			batch_count += 1

		print "epoch",str(k+1),"; loss :",(sum_cost/batch_count),"; accuracy :",(sum_accuracy * 100 /batch_count),"% ; time :",(time.time() - start_time),"sec"

	print "Training Complete"

	saver.save(sess, path)

def test():

	path = "weights/model_simple.ckpt"
	if os.path.isfile((path+".meta")):
		print "Loading the model..."
		saver = tf.train.Saver()
		savePath = saver.restore(sess, path)
		print "Loading complete"
	else:
		print "Error model never trained before"
		return 1

	pickle_in = open("processed_data/test_images_processed.pickle","rb")
	test_images_processed = pickle.load(pickle_in)
	pickle_in.close()

	p = test_images_processed.keys()

	img = mpimg.imread("data/Flickr8k_Dataset/"+p[0])
	imgplot = plt.imshow(img)
	plt.show()

	img = np.reshape(test_images_processed[p[0]],[-1,64,64])

	images_batch = []
	for i in range(0,len(img[0])):
		temp = []
		for j in range(0,len(img)):
			temp.append(img[j][i])
		images_batch.append(temp)

	pickle_in = open("processed_data/word2vec.pickle","rb")
	word2vec = pickle.load(pickle_in)
	pickle_in.close()

	pickle_in = open("processed_data/vec2word.pickle","rb")
	vec2word = pickle.load(pickle_in)
	pickle_in.close()

	start_word = ["<start>"]
	while True:
	    par_caps = [word2vec[i] for i in start_word]
	    par_caps = tf.keras.preprocessing.sequence.pad_sequences([par_caps], maxlen=64, padding='post')

	    preds = sess.run([nn.prediction],feed_dict={nn.encoder_emb_inp:images_batch,nn.decoder_emb_inp:np.array(par_caps)})

	    word_pred = vec2word[np.argmax(preds[0])]
	    start_word.append(word_pred)
	        
	    if word_pred == "<end>" or len(start_word) > 64:
	        break
	            
	Argmax_Search = ' '.join(start_word[1:-1])

	print Argmax_Search



##################################
#								 #
# main code for argument parsing #
#								 #
##################################

parser = argparse.ArgumentParser(description='RNN Training')
# Add arguments
parser.add_argument('--train', help='train or test is required',action='store_true')
parser.add_argument('--test', help='train or test is required',action='store_true')
# Array for all arguments passed to script
args = parser.parse_args()
if not args.train and not args.test:
   	parser.error ('Either --train or --test is required.')

nn = NNModel()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	if args.train:
	   	train()
	else:
	    test()