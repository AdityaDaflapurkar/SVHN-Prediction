from __future__ import print_function
import numpy as np
import tensorflow as tf


def weight_variable(shape, nameVar):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=nameVar)

        # creates and returns a bias variable with given shape initialized with
        # a constant of 0.1
def bias_variable(shape, nameVar):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=nameVar)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])

print("Success")

image_size = 32
num_labels = 11 # 0-9, + blank 
num_channels = 1 # grayscale

batch_size = 64
patch_size = 5
depth1 = 16
depth2 = 32
#depth3 = 64
num_hidden1 = 64
#num_hidden2 = 16
shape = [batch_size, image_size, image_size, num_channels]


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
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = weight_variable([patch_size, patch_size, num_channels, depth1] , "W1")
  layer1_biases = bias_variable([depth1] , 'B1')
  layer2_weights = weight_variable([patch_size, patch_size, depth1, depth2] , "W2")
  layer2_biases = bias_variable( [depth2], 'B2')
  layer3_weights = weight_variable([patch_size, patch_size, depth2, num_hidden1] , "W3")
  layer3_biases = bias_variable([num_hidden1], 'B3')

  s1_w = tf.weight_variable( [num_hidden1, num_labels],"WS1")
  s1_b = tf.bias_variable([num_labels], 'BS1')
  s2_w = tf.weight_variable([num_hidden1, num_labels] , "WS2")
  s2_b = tf.bias_variable([num_labels], 'BS2')
  s3_w = tf.weight_variable( [num_hidden1, num_labels] , "WS3")
  s3_b = tf.bias_variable([num_labels], 'BS3')
  s4_w = tf.weight_variable([num_hidden1, num_labels] , "WS4")
  s4_b = tf.bias_variable([num_labels], 'BS4')
  s5_w = tf.weight_variable( [num_hidden1, num_labels] , "WS5")
  s5_b = tf.bias_variable([num_labels], 'BS5')


def model(data, keep_prob, shape):
	h_conv1 = tf.nn.relu(conv2d(data, layer1_weights) + layer1_biases)
    #lrn = tf.nn.local_response_normalization(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, layer2_weights) + layer2_biases)
    #lrn = tf.nn.local_response_normalization(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    shape = h_pool2.get_shape().as_list()
    h_pool2_flat = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])

    # conv = tf.nn.conv2d(sub, layer3_weights, [1,1,1,1], padding='VALID', name='C5')
    # hidden = tf.nn.relu(conv + layer3_biases)

	W_fc1 = weight_variable([7 * 7 * 64, size_fully_connected_layer], "W_fc1")
	b_fc1 = bias_variable([size_fully_connected_layer], "b_fc1")
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    shape1 = hidden.get_shape().as_list()
    reshape = tf.reshape(h_fc1_drop, [shape1[0], shape1[1] * shape1[2] * shape1[3]])

    logits1 = tf.matmul(reshape, s1_w) + s1_b
    logits2 = tf.matmul(reshape, s2_w) + s2_b
    logits3 = tf.matmul(reshape, s3_w) + s3_b
    logits4 = tf.matmul(reshape, s4_w) + s4_b
    logits5 = tf.matmul(reshape, s5_w) + s5_b
    return [logits1, logits2, logits3, logits4, logits5]

# Training computation.
[logits1, logits2, logits3, logits4, logits5] = model(tf_train_dataset, 0.9, shape)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits1, tf_train_labels[:,1])) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, tf_train_labels[:,2])) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits3, tf_train_labels[:,3])) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits4, tf_train_labels[:,4])) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits5, tf_train_labels[:,5]))


#Opittmizer
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

# Predictions
train_prediction = tf.pack([tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[0]),\
                      tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[1]),\
                      tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[2]),\
                      tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[3]),\
                      tf.nn.softmax(model(tf_train_dataset, 1.0, shape)[4])])
valid_prediction = tf.pack([tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[0]),\
                      tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[1]),\
                      tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[2]),\
                      tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[3]),\
                      tf.nn.softmax(model(tf_valid_dataset, 1.0, shape)[4])])
test_prediction = tf.pack([tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[0]),\
                     tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[1]),\
                     tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[2]),\
                     tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[3]),\
                     tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[4])])


#saver = tf.train.Saver()


num_steps = 5000
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()  
  
  "to be used after first iteration"
  #saver.restore(session, "CNN_multi.ckpt")
  #print("Model restored.") 

  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size),:]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    
    if (step % 500 == 0): 
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels[:,1:6]))
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels[:,1:6]))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels[:,1:6]))
  save_path = saver.save(session, "CNN_multi.ckpt")
  print("Model saved in file: %s" % save_path)



