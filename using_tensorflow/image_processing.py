'''

Program to create processed image data from pretrained VGG16

'''
import os
import pickle 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

path_images 				= 	"data/Flicker8k_Dataset/"
path_captions 				= 	"data/Flickr8k.token.txt"
train_data_path 			= 	"data/Flickr_8k.trainImages.txt"
test_data_path 				= 	"data/Flickr_8k.testImages.txt"
processed_data_directory 	= 	"processed_data"

model = tf.keras.applications.VGG16(include_top=True,weights='imagenet',input_tensor=None,input_shape=None,pooling=None,classes=1000)
model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

train_images_processed = {}
test_images_processed = {}

def process_image(filename):

	filename = "data/Flickr8k_Dataset/"+filename
	image = tf.contrib.keras.preprocessing.image.load_img(filename, target_size=(224, 224))

	array = tf.contrib.keras.preprocessing.image.img_to_array(image)
	array = np.expand_dims(array, axis=0)

	image = model.predict(array)[0]

	return image

#create train data

print "Processing Train Images..."

fd = open(train_data_path,"r")

for line in fd.readlines():
	image 	= line.rstrip()
	output 	= process_image(image)
	train_images_processed[image] = output

pickle_out = open("processed_data/train_images_processed.pickle","wb")
pickle.dump(train_images_processed, pickle_out)
pickle_out.close()

print "Processing Train Images complete..."

#end of train data

#create test data

print "Processing Test Images..."

fd = open(test_data_path,"r")

for line in fd.readlines():
	image 	= line.rstrip()
	output 	= process_image(image)
	test_images_processed[image] = output
	
pickle_out = open("processed_data/test_images_processed.pickle","wb")
pickle.dump(test_images_processed, pickle_out)
pickle_out.close()

print "Processing Test Images complete..."

#end of test data