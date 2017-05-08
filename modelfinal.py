import numpy as np
import tensorflow as tf
from skimage.io import imread_collection,imshow
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import gzip

class NN_Model():

    path = ""

    def __init__(self):
		
		#self.graph=tf.Graph()
		#with self.graph.as_default():
		#config = tf.ConfigProto(allow_soft_placement = True)
			
		
			image_size = 32
			num_labels = 11 # 0-9, + blank 
			num_channels = 3 # grayscale
			batch_size = 1
			patch_size = 5
			depth1 = 16
			depth2 = 32
			depth3 = 64
			num_hidden1 = 64
			shape = [batch_size, image_size, image_size, num_channels]	
			final_image_size = self.output_size_pool(input_size=image_size,conv_filter_size=5, pool_filter_size=2,padding='valid',\
				conv_stride=1,pool_stride=2) 

			"""define weights and biases"""
	  		# Input data.
			self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
			self.tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, 6))
			self.tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
				#tf_test_dataset = tf.constant(test_dataset)
	
	
		# Variables.
			self.layer1_weights = self.weight_variable([patch_size, patch_size, num_channels, depth1] , "W1")
			self.layer1_biases = self.bias_variable([depth1] , 'B1')
			self.layer2_weights = self.weight_variable([patch_size, patch_size, depth1, depth2] , "W2")
			self.layer2_biases = self.bias_variable( [depth2], 'B2')
				#shape11 = layer2_weights.get_shape().as_list()
	
				# layer3_weights = weight_variable([patch_size, patch_size, depth2, num_hidden1] , "W3")
				# layer3_biases = bias_variable([num_hidden1], 'B3')

			self.layer3_weights = self.weight_variable([patch_size, patch_size, depth2, num_hidden1] , "W3")
			self.layer3_biases = self.bias_variable([num_hidden1], 'B3')

			self.w_fc1 = self.weight_variable([final_image_size*final_image_size*depth3, num_hidden1], 'w_fc1')
			self.b_fc1 = self.bias_variable([num_hidden1], 'b_fc1')

			self.s1_w = self.weight_variable( [num_hidden1, num_labels],"WS1")
			self.s1_b = self.bias_variable([num_labels], 'BS1')
			self.s2_w = self.weight_variable([num_hidden1, num_labels] , "WS2")
			self.s2_b = self.bias_variable([num_labels], 'BS2')
			self.s3_w = self.weight_variable( [num_hidden1, num_labels] , "WS3")
			self.s3_b = self.bias_variable([num_labels], 'BS3')
			self.s4_w = self.weight_variable([num_hidden1, num_labels] , "WS4")
			self.s4_b = self.bias_variable([num_labels], 'BS4')
			self.s5_w = self.weight_variable( [num_hidden1, num_labels] , "WS5")
			self.s5_b = self.bias_variable([num_labels], 'BS5')
		
			[logits1, logits2, logits3, logits4, logits5] = self.model(0.9,shape)
			self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=self.tf_train_labels[:,1])) +\
				tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=self.tf_train_labels[:,2])) +\
				tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3, labels=self.tf_train_labels[:,3])) +\
				tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits4, labels=self.tf_train_labels[:,4])) +\
				tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits5, labels=self.tf_train_labels[:,5]))
			tf.summary.scalar('loss', self.loss)
			self.merged = tf.summary.merge_all()
		
			# Optimizer
			self.global_step = tf.Variable(0)
			self.learning_rate = tf.train.exponential_decay(0.05,self.global_step, 10000, 0.95)
			self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
		
			# Predictions
			self.train_prediction = tf.stack([tf.nn.softmax(self.model(1.0,shape)[0]),\
		                      tf.nn.softmax(self.model(1.0,shape)[1]),\
		                      tf.nn.softmax(self.model(1.0,shape)[2]),\
		                      tf.nn.softmax(self.model(1.0,shape)[3]),\
		                      tf.nn.softmax(self.model(1.0,shape)[4])])
			
			self.session = tf.Session()#(graph=self.graph)
			self.session.run(tf.global_variables_initializer())
			self.saver = tf.train.Saver()


    def weight_variable(self, shape, nameVar):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name=nameVar)


    def bias_variable(self, shape, nameVar):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, name=nameVar)


    def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


    def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    def accuracy(self, predictions, labels):
		print labels
		return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])

	
    def output_size_pool(self, input_size, conv_filter_size, pool_filter_size,padding, conv_stride, pool_stride):
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
		output_5 = (((output_4 - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
		# After pool 2
		#	output_6 = (((output_5 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
		return int(output_5)


    def model(self, keep_prob, shape):
		data=self.tf_train_dataset
		h_conv1 = tf.nn.relu(self.conv2d(data, self.layer1_weights) + self.layer1_biases)
		#lrn = tf.nn.local_response_normalization(h_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.layer2_weights) + self.layer2_biases)
		#lrn = tf.nn.local_response_normalization(h_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)

		h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.layer3_weights) + self.layer3_biases)
		#lrn = tf.nn.local_response_normalization(h_conv2)
		# h_pool3 = max_pool_2x2(h_conv3)

		h_pool2_drop = tf.nn.dropout(h_conv3, keep_prob)
		shape = h_pool2_drop.get_shape().as_list()
		h_pool2_flat = tf.reshape(h_pool2_drop, [shape[0], shape[1] * shape[2] * shape[3]])



		# conv = tf.nn.conv2d(sub, layer3_weights, [1,1,1,1], padding='VALID', name='C5')
	    # hidden = tf.nn.relu(conv + layer3_biases)
	#	W_fc1 = weight_variable([(shape[1] * shape[2] * shape[3]), size_fully_connected_layer], "W_fc1")
	#	b_fc1 = bias_variable([size_fully_connected_layer], "b_fc1")
		reshape = tf.nn.relu(tf.matmul(h_pool2_flat, self.w_fc1) + self.b_fc1)

	#	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	#	shape1 = h_fc1_drop.get_shape().as_list()
	#	reshape = tf.reshape(h_fc1_drop, [shape1[1], shape1[0]])

		logits1 = tf.matmul(reshape, self.s1_w) + self.s1_b
		print self.s1_w.get_shape(),"sssssssssss"
		print np.shape(logits1),"xxxxxxxxx"
		logits2 = tf.matmul(reshape, self.s2_w) + self.s2_b
		logits3 = tf.matmul(reshape, self.s3_w) + self.s3_b
		logits4 = tf.matmul(reshape, self.s4_w) + self.s4_b
		logits5 = tf.matmul(reshape, self.s5_w) + self.s5_b
		return [logits1, logits2, logits3, logits4, logits5]

    def train(self):
		df=pd.read_csv("svhn.csv")
		print "Collecting images..."	
		imgset = imread_collection("data/train_images/*.png")

	
		train_dataset = imgset
		train_labels = df.ix[:,4:10]

		summary_writer = tf.summary.FileWriter("logs_final")#, graph=self.graph)
	
		num_steps = 33402
		print('Initialized')
	
		for i in xrange(6):#150
       			
			for step in range(5567,num_steps):

				crop=train_dataset[step][ df.get_value(step,"y1"):df.get_value(step,"y2"),df.get_value(step,"x1"):df.get_value(step,"x2"),:]
				resized_image=resize(crop, (32,32),mode='reflect')
		
				rs=[]
				rs.append(resized_image)
				batch_data = rs
				batch_labels = np.reshape(train_labels.ix[step],(1,len(train_labels.ix[step])))
	
				feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
				_, l, predictions, summary = self.session.run([self.optimizer, self.loss, self.train_prediction,self.merged],feed_dict=feed_dict)
		
				if step%1000==0:
					print(i,' : Minibatch loss at step %d: %f' % (step, l))
					print np.argmax(predictions,2).T
					print('Minibatch accuracy: %.1f%%' % self.accuracy(predictions, batch_labels[:,1:6]))
				summary_writer.add_summary(summary, i*num_steps+step)
		


    def get_sequence(self, image):
		resized_image=resize(image, (32,32),mode='reflect')
		rs=[]
		rs.append(resized_image)
		feed_dict = {self.tf_train_dataset : rs}
		predictions = self.session.run(self.train_prediction, feed_dict=feed_dict)
		predicted=np.argmax(predictions,2).T
		for i in xrange(len(predicted[0])):
			if predicted[0][i]==10:
				break
		return predicted[0][:i]


    def save_model(self, ckpt_dir):
		
		save_path = self.saver.save(self.session, ckpt_dir+"model.ckpt")
   		print("Saved to path: ", save_path)

        
    @staticmethod
    def load_model(ckpt_dir):
		model = NN_Model()
#		session = tf.Session()
		saver=tf.train.Saver()
		saver.restore(model.session, ckpt_dir+"model.ckpt")		
		#model.session = session
		print "Model loaded successfully!!"
		return model



if __name__ == "__main__":
	#obj = NN_Model()
	#obj.train()
	#obj.save_model("checkpoint_final/model.ckpt")
	
	
	obj = NN_Model.load_model("checkpoint_final/")

	df=pd.read_csv("svhn.csv")
	print "Collecting images..."	
	imgset = imread_collection("data/train_images/*.png")

	
	train_dataset = imgset
	train_labels = df.ix[:,5:10]
	
	
	# Test
	corr=0
	for step in range(1, 5567):

		crop=train_dataset[step][ df.get_value(step,"y1"):df.get_value(step,"y2"),df.get_value(step,"x1"):df.get_value(step,"x2"),:]
		# resized_image=resize(crop, (32,32),mode='reflect')
		

		predicted = obj.get_sequence(crop)
		labels = np.reshape(train_labels.ix[step],(1,len(train_labels.ix[step])))
		print predicted,"predicted"
		print labels[0][:len(predicted)],"labels"
		
		if np.array_equal(predicted,labels[0][:len(predicted)]):
			corr=corr+1
	print corr*100.0/5567


	
