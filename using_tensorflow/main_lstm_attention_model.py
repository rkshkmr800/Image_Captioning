import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import pickle
import time
import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

tf.set_random_seed(2)

num_units = 128
learning_rate = 0.001
epochs = 50
batch_size = 512

pickle_in = open("processed_data/max_length.pickle","rb")
max_length = pickle.load(pickle_in)
pickle_in.close() 

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

source_sequence_length = tf.constant(4096)

max_length = 64
embedding_size = 64

encoder_emb_inp = tf.placeholder(tf.float32,[64,None,64])
decoder_inputs = tf.placeholder(tf.int32,[None,None])
decoder_outputs = tf.placeholder(tf.float32,[None,vocabulary_size])
source_sequence_length = tf.placeholder(tf.int32,[None,1])

# Embedding
embedding_decoder = tf.get_variable("embedding_decoder", [vocabulary_size, embedding_size])
# Look up embedding:
#   decoder_inputs: [max_time, batch_size]
#   decoder_emb_inp: [max_time, batch_size, embedding_size]
decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, tf.transpose(decoder_inputs))

# Build RNN cell
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Run Dynamic RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, dtype=tf.float32, time_major=True)


attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
#attention_states = encoder_outputs
# Create an attention mechanism
attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_states,memory_sequence_length=source_sequence_length)


# Build RNN cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=num_units)

initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
initial_state = initial_state.clone(cell_state=encoder_state)

# Helper
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, [max_length], time_major=True)
# Decoder

#decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
# Dynamic decoding

outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)

logits =  tf.layers.dense(outputs.rnn_output[-1], vocabulary_size)

prediction = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits))

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(decoder_outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


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
		if images_count == 2000:
			break
	

	path = "weights/model_lstm_attention_model.ckpt"
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
			lengths = []
			for t in range(0,batch_size):
				cap = []
				for i in captions[j]:
					cap.append(i)
				cap = np.trim_zeros(cap, 'b')
				lengths.append(len(cap))
				captions_batch.append(cap)


			_,loss,acc = sess.run([optimizer,cost,accuracy],feed_dict={source_sequence_length:lengths,encoder_emb_inp:images_batch,decoder_inputs:captions_batch,decoder_outputs:next_words_batch})
			sum_cost += loss
			sum_accuracy += acc
			batch_count += 1

		print "epoch",str(k+1),"; loss :",(sum_cost/batch_count),"; accuracy :",(sum_accuracy * 100 /batch_count),"% ; time :",(time.time() - start_time),"sec"

	print "Training Complete"

	savePath = saver.save(sess, path)

	print "Saved the model...path :",savePath

def test():

	path = "weights/model_lstm_attention_model.ckpt"
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

	    preds = sess.run([prediction],feed_dict={encoder_emb_inp:images_batch,decoder_inputs:np.array(par_caps)})

	    word_pred = vec2word[np.argmax(preds[0])]
	    start_word.append(word_pred)
	        
	    if word_pred == "<end>" or len(start_word) > 40:
	        break
	            
	Argmax_Search = ' '.join(start_word[1:-1])

	print Argmax_Search
	plt.show()


##################################
#				 #
# main code for argument parsing #
#				 #
##################################

parser = argparse.ArgumentParser(description='RNN Training')
# Add arguments
parser.add_argument('--train', help='train or test is required',action='store_true')
parser.add_argument('--test', help='train or test is required',action='store_true')
# Array for all arguments passed to script
args = parser.parse_args()
if not args.train and not args.test:
   	parser.error ('Either --train or --test is required.')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	if args.train:
	   	train()
	else:
		test()
