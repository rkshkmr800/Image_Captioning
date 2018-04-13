'''

Program to create processed train and test data, create word2vec and vec2word mappings

'''
import os
import pickle 

path_images 				= 	"data/Flicker8k_Dataset/"
path_captions 				= 	"data/Flickr8k.token.txt"
train_data_path 			= 	"data/Flickr_8k.trainImages.txt"
test_data_path 				= 	"data/Flickr_8k.testImages.txt"
processed_data_directory 	= 	"processed_data"

if not os.path.exists(processed_data_directory):
    os.makedirs(processed_data_directory)

#processing all the capitons related to images to dictionary

images_with_caption = {}

fd = open(path_captions,"r")

for line in fd.readlines():
	line = line.split("#")
	image = line[0]
	caption = line[1].split("\t")[1]
	caption = caption.rstrip()
	caption = "<start> "+caption+" <end>"
	if image in images_with_caption:
		images_with_caption[image].append(caption)
	else:
		images_with_caption[image] = [caption]

fd.close()

pickle_out = open("processed_data/images_with_caption.pickle","wb")
pickle.dump(images_with_caption, pickle_out)
pickle_out.close()

#end of processing

#create train data

fd = open(train_data_path,"r")

trainfd = open("processed_data/train_data.txt","w")

train_images_with_caption = {}

for line in fd.readlines():
	image = line.rstrip()
	train_images_with_caption[image] = images_with_caption[image]
	for caption in images_with_caption[image]:
		trainfd.write(image+" | "+caption+"\n")

fd.close()
trainfd.close()

pickle_out = open("processed_data/train_images_with_caption.pickle","wb")
pickle.dump(train_images_with_caption, pickle_out)
pickle_out.close()

#end of train data

#create test data

fd = open(test_data_path,"r")

testfd = open("processed_data/test_data.txt","w")

test_images_with_caption = {}

for line in fd.readlines():
	image = line.rstrip()
	test_images_with_caption[image] = images_with_caption[image]
	for caption in images_with_caption[image]:
		testfd.write(image+" | "+caption+"\n")

fd.close()
testfd.close()

pickle_out = open("processed_data/test_images_with_caption.pickle","wb")
pickle.dump(test_images_with_caption, pickle_out)
pickle_out.close()


#end of test data

#finding the vocabulary

train_fd = open("processed_data/train_data.txt","r")

total_words = []
max_length = 0

for line in train_fd.readlines():

	line = line.rstrip()
	line = line.split(" | ")
	caption = line[1].rstrip()
	caption = caption.split()

	length = len(caption)
	if length > max_length:
		max_length = length

	total_words.extend(caption) 

total_words = list(set(total_words))
vocabulary_size = len(total_words)

print "max_length of senetence (word_count) : ",max_length
print "vocabulary size : ",vocabulary_size

pickle_out = open("processed_data/max_length.pickle","wb")
pickle.dump(max_length, pickle_out)
pickle_out.close()

pickle_out = open("processed_data/vocabulary_size.pickle","wb")
pickle.dump(vocabulary_size, pickle_out)
pickle_out.close()

#end of block

#creating word2vec and vec2word for the train dataset

word2vec = {}
vec2word = {}

for i in range(0,vocabulary_size):
	word2vec[total_words[i]] = i 
	vec2word[i] = total_words[i]

pickle_out = open("processed_data/word2vec.pickle","wb")
pickle.dump(word2vec, pickle_out)
pickle_out.close()
print "word2vec mapping saved..."

pickle_out = open("processed_data/vec2word.pickle","wb")
pickle.dump(vec2word, pickle_out)
pickle_out.close()
print "vec2word mapping saved..."

#end of block