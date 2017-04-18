import numpy as np
import tensorflow as tf
from skimage.io import imread_collection,imshow
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("svhn.csv")
print "Collecting images..."	
imgset = imread_collection("train_images/*.png")

'''
for i in xrange(len(images)):
		print i+1			
		crop=images[i][ df.get_value(i,"y1"):df.get_value(i,"y2"),df.get_value(i,"x1"):df.get_value(i,"x2"),:]
		resized_image=resize(crop, (32,32),mode='reflect')
		imgset.append(resized_image)
'''
#print imgset[0]#," images collected."

# pickle_file = 'SVHN_multi.pickle'

# with open(pickle_file, 'rb') as f:
#   save = pickle.load(f)
train_dataset = imgset
train_labels = df.ix[:,4:10]
print len(train_dataset),"aaa ",len(train_labels)
valid_dataset = imgset[15000:]
valid_labels = df.ix[15000:,4:10]
print len(valid_dataset),"bbb ",len(valid_labels)
#   test_dataset = save['test_dataset']
#   test_labels = save['test_labels']
#   del save  
#   print('Training set', train_dataset.shape, train_labels.shape)
#   print('Validation set', valid_dataset.shape, valid_labels.shape)
#   print('Test set', test_dataset.shape, test_labels.shape)

def weight_variable(shape, nameVar):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=nameVar)

        # creates and returns a bias variable with given shape initialized with
        # a constant of 0.1
def bias_variable(shape, nameVar):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=nameVar)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def accuracy(predictions, labels):
	print labels
	return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])

def output_size_pool(input_size, conv_filter_size, pool_filter_size,padding, conv_stride, pool_stride):
	if padding == 'same':
		padding = -1.00
	elif padding == 'valid':
		padding = 0.00
	else:
		return None
	#after conv 1
	output_1 = (((input_size - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
	# After pool 1
	output_2 = (((output_1 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
	# After convolution 2
	output_3 = (((output_2 - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
	# After pool 2
	output_4 = (((output_3 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
	# After convolution 2
#	output_5 = (((output_4 - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
	# After pool 2
#	output_6 = (((output_5 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
	return int(output_4)


print("Success")

image_size = 32
num_labels = 11 # 0-9, + blank 
num_channels = 3 # grayscale

batch_size = 1
patch_size = 5
depth1 = 16
depth2 = 32
#depth3 = 64
num_hidden1 = 64
#num_hidden2 = 16
shape = [batch_size, image_size, image_size, num_channels]
size_fully_connected_layer = 11	
final_image_size = output_size_pool(input_size=image_size,conv_filter_size=5, pool_filter_size=2,padding='valid', conv_stride=1,pool_stride=2) 


# 7-layer CNN.
# C1: convolutional layer, batch_size x 28 x 28 x 16, convolution size: 5 x 5 x 1 x 16
# S2: sub-sampling layer, batch_size x 14 x 14 x 16
# C3: convolutional layer, batch_size x 10 x 10 x 32, convolution size: 5 x 5 x 16 x 32
# S4: sub-sampling layer, batch_size x 5 x 5 x 32
# C5: convolutional layer, batch_size x 1 x 1 x 64, convolution size: 5 x 5 x 32 x 64
# Dropout
# F6: fully-connected layer, weight size: 64 x 16
# Output layer, weight size: 16 x 10

graph = tf.Graph()
with graph.as_default():
	"""define weights and biases"""
  # Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, 6))
	tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	#tf_test_dataset = tf.constant(test_dataset)


	# Variables.
	layer1_weights = weight_variable([patch_size, patch_size, num_channels, depth1] , "W1")
	layer1_biases = bias_variable([depth1] , 'B1')
	layer2_weights = weight_variable([patch_size, patch_size, depth1, depth2] , "W2")
	layer2_biases = bias_variable( [depth2], 'B2')
	shape11 = layer2_weights.get_shape().as_list()

	# layer3_weights = weight_variable([patch_size, patch_size, depth2, num_hidden1] , "W3")
	# layer3_biases = bias_variable([num_hidden1], 'B3')

	w_fc1 = weight_variable([final_image_size*final_image_size*depth2, num_hidden1], 'w_fc1')
	b_fc1 = bias_variable([num_hidden1], 'b_fc1')

	s1_w = weight_variable( [num_hidden1, num_labels],"WS1")
	s1_b = bias_variable([num_labels], 'BS1')
	s2_w = weight_variable([num_hidden1, num_labels] , "WS2")
	s2_b = bias_variable([num_labels], 'BS2')
	s3_w = weight_variable( [num_hidden1, num_labels] , "WS3")
	s3_b = bias_variable([num_labels], 'BS3')
	s4_w = weight_variable([num_hidden1, num_labels] , "WS4")
	s4_b = bias_variable([num_labels], 'BS4')
	s5_w = weight_variable( [num_hidden1, num_labels] , "WS5")
	s5_b = bias_variable([num_labels], 'BS5')


	def model(data, keep_prob, shape):
		h_conv1 = tf.nn.relu(conv2d(data, layer1_weights) + layer1_biases)
		#lrn = tf.nn.local_response_normalization(h_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(conv2d(h_pool1, layer2_weights) + layer2_biases)
		#lrn = tf.nn.local_response_normalization(h_conv2)
		h_pool2 = max_pool_2x2(h_conv2)
		h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)
		shape = h_pool2_drop.get_shape().as_list()
		h_pool2_flat = tf.reshape(h_pool2_drop, [shape[0], shape[1] * shape[2] * shape[3]])

		# conv = tf.nn.conv2d(sub, layer3_weights, [1,1,1,1], padding='VALID', name='C5')
	    # hidden = tf.nn.relu(conv + layer3_biases)
	#	W_fc1 = weight_variable([(shape[1] * shape[2] * shape[3]), size_fully_connected_layer], "W_fc1")
	#	b_fc1 = bias_variable([size_fully_connected_layer], "b_fc1")
		reshape = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

	#	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	#	shape1 = h_fc1_drop.get_shape().as_list()
	#	reshape = tf.reshape(h_fc1_drop, [shape1[1], shape1[0]])

		logits1 = tf.matmul(reshape, s1_w) + s1_b
		logits2 = tf.matmul(reshape, s2_w) + s2_b
		logits3 = tf.matmul(reshape, s3_w) + s3_b
		logits4 = tf.matmul(reshape, s4_w) + s4_b
		logits5 = tf.matmul(reshape, s5_w) + s5_b
		return [logits1, logits2, logits3, logits4, logits5]

	# Training computation.
	[logits1, logits2, logits3, logits4, logits5] = model(tf_train_dataset, 0.9, shape)

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=tf_train_labels[:,1])) +\
	tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=tf_train_labels[:,2])) +\
	tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3, labels=tf_train_labels[:,3])) +\
	tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits4, labels=tf_train_labels[:,4])) +\
	tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits5, labels=tf_train_labels[:,5]))
	tf.summary.scalar('loss', loss)
	merged = tf.summary.merge_all()

	#Optimizer
	global_step = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
	optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

	# Predictions
	train_prediction = tf.stack([tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[0]),\
	                      tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[1]),\
	                      tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[2]),\
	                      tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[3]),\
	                      tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[4])])
	
	valid_prediction = tf.stack([tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[0]),\
	                      tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[1]),\
	                      tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[2]),\
	                      tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[3]),\
	 	                  tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[4])])
	
	'''	
	test_prediction = tf.stack([tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[0]),\
	                     tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[1]),\
	                     tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[2]),\
	                     tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[3]),\
	                     tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[4])])
	'''

#saver = tf.train.Saver()



with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()

	summary_writer = tf.summary.FileWriter("/home/aditya/Downloads/House_number_prediction/logs", graph=tf.get_default_graph())

	num_steps = 27835
	print('Initialized')
	
	for i in xrange(1):#150
       	  for step in range(num_steps):
		#offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		#print (train_dataset[step])
		
		crop=train_dataset[step][ df.get_value(step,"y1"):df.get_value(step,"y2"),df.get_value(step,"x1"):df.get_value(step,"x2"),:]
		resized_image=resize(crop, (32,32),mode='reflect')
		
		rs=[]
		rs.append(resized_image)
		batch_data = rs
		batch_labels = np.reshape(train_labels.ix[step],(1,len(train_labels.ix[step])))
		
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions, summary = session.run([optimizer, loss, train_prediction, merged], feed_dict=feed_dict)
		

		if step%1000==0:
			print(i,' : Minibatch loss at step %d: %f' % (step, l))
			print np.argmax(predictions,2).T
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels[:,1:6]))
			summary_writer.add_summary(summary, step)
	
	print "Validation:"
	
	for step in range(0,1):#(27835,28000):
		
		crop=train_dataset[step][ df.get_value(step,"y1"):df.get_value(step,"y2"),df.get_value(step,"x1"):df.get_value(step,"x2"),:]
		resized_image=resize(crop, (32,32),mode='reflect')
		#print crop
		imshow(crop)
		plt.show()
		rs=[]
		rs.append(resized_image)
		batch_data = rs
		batch_labels = np.reshape(train_labels.as_matrix()[step],(1,len(train_labels.as_matrix()[step])))
		
		feed_dict = {tf_train_dataset : batch_data}
		predictions = session.run([train_prediction,], feed_dict=feed_dict)
		
		p=np.reshape(predictions,[5,1,11])
		print "v.p.",np.argmax(p,2).T
		print "v.p.",batch_labels[:,1:]
		#print(i,' : Minibatch loss at step %d: %f' % (step, l))
		#print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels[:,1:6]))
		
	

	saver = tf.train.Saver()
	save_path = saver.save(session, "/home/aditya/Downloads/House_number_prediction/models/model.ckpt")
   	print("Saved to path: ", save_path)
       	
	
	#print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels[:,1:6]))
	#print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels[:,1:6]))
	#save_path = saver.save(session, "CNN_multi.ckpt")
	#print("Model saved in file: %s" % save_path)

