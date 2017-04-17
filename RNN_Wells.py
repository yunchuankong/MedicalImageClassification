# from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
import RNN_util

reset_default_graph()

# Parameters
batch_size = 50
# display_step = 500
# n_embedding = 1000 # ?
hidden_layer_size = 1000
input_size = 4096 # n_features
target_size = 2 # n_classes
learning_rate = 0.001
reg_scale = 0.00001
training_iters = 10000

def data_summary(data):
    summary = dict()
    summary["len"] = len(data)
    summary["n_slices"] = np.array(list(map(lambda x: len(x[0]), data)))
    summary["n_features"] = input_size
    summary["max_seq_len"] = summary["n_slices"].max()
    summary["cancer"] = np.array(list(map(lambda x: x[1][1], data)))
    return summary

class DataBatchGenerator(object):
    """
    This class generate samples for training and testing (random sampling with a given proportion):
    - Class 0: normal samples
    - Class 1: cancer samples
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """

    def __init__(self, data, summary, test_prop=0.1,batch_size=100):
        self.images = []
        self.labels = []
        self.seqlen = summary["n_slices"]
        max_seq_len = summary["max_seq_len"]
        for i in range(summary["len"]):
            self.images.append(np.concatenate((data[i][0], np.zeros([max_seq_len - self.seqlen[i], input_size]))))
            # pad each sequence to reach 'max_seq_len'
            self.labels.append(data[i][1])
        # Obtain the training set and test set for RNN
        self.test = np.random.choice(summary["len"], int(round(test_prop * summary["len"])), replace=False)
        self.train = np.setdiff1d(np.array(range(summary["len"])), self.test)
        self.batch_size=batch_size

    def next_batch(self,batch_size,batch_id):
        len_train = len(self.train)
        n_bins = len_train / batch_size
        batch_images = []
        batch_labels = []
        batch_seqlen = []
        if batch_id > (n_bins+1):
            print("batch_id too large!")
            return None
        if batch_id == n_bins+1:
            idx = self.train[batch_size * (batch_id - 1): len_train]
            for id in idx:
                batch_images.append(self.images[id])
                batch_labels.append(self.labels[id])
                batch_seqlen.append(self.seqlen[id])
            extr_batch_images, extr_batch_labels, extr_batch_seqlen = \
                self.next_random_batch(batch_size-len(idx), train_exclude=np.setdiff1d(self.train,idx))
            batch_images.append(extr_batch_images)
            batch_labels.append(extr_batch_labels)
            batch_seqlen.append(extr_batch_seqlen)
            return batch_images, batch_labels, batch_seqlen

        idx = self.train[ batch_size * (batch_id-1): (batch_size * batch_id - 1) ]
        for id in idx:
            batch_images.append(self.images[id])
            batch_labels.append(self.labels[id])
            batch_seqlen.append(self.seqlen[id])
        return batch_images, batch_labels, batch_seqlen

    def next_random_batch(self, batch_size, train_exclude=None):
        """ Return a batch of data randomly from training set.
        """
        if train_exclude.any():
            ids = np.setdiff1d(self.train,train_exclude)
            len_choice = len(ids)
        else:
            len_choice = len(self.train)
        batch_id = np.random.choice(len_choice, batch_size, replace=False)
        batch_images = []
        batch_labels = []
        batch_seqlen = []
        for i in batch_id:
            idx = self.train[i]
            batch_images.append(self.images[idx])
            batch_labels.append(self.labels[idx])
            batch_seqlen.append(self.seqlen[idx])

        return batch_images, batch_labels, batch_seqlen

    def get_train(self):
        """Return the whole training dataset.
        """
        train_images = []
        train_labels = []
        train_seqlen = []
        for i in self.train:
            train_images.append(self.images[i])
            train_labels.append(self.labels[i])
            train_seqlen.append(self.seqlen[i])

        return train_images, train_labels, train_seqlen

    def get_test(self):
        """Return the selected testing dataset.
        """
        test_images = []
        test_labels = []
        test_seqlen = []
        for i in self.test:
            test_images.append(self.images[i])
            test_labels.append(self.labels[i])
            test_seqlen.append(self.seqlen[i])

        return test_images, test_labels, test_seqlen


# ==========
#   MODEL
# ==========

bottle_neck_data = np.load(
    "/labs/colab/3DDD/kaggle_data/kaggle_processed_data/resampled_transfer_learning_data_new.npy")
bottle_neck_summary = data_summary(bottle_neck_data)
dataset = DataBatchGenerator(bottle_neck_data, bottle_neck_summary, 0.10, 100)
seq_max_len = bottle_neck_summary["max_seq_len"]

y = tf.placeholder(tf.float32, shape=[None, target_size], name='inputs')

# # Models
# Initializing rnn object
rnn = RNN_util.LSTM_cell(input_size, hidden_layer_size, target_size)
# In[6]:
# Getting all outputs from rnn
outputs = rnn.get_outputs()
# Getting final output through indexing after reversing
last_output = tf.reverse(outputs, [True, False, False])[0, :, :]

# As rnn model output the final layer through Relu activation softmax is
# used for final output.
output = tf.nn.softmax(last_output)
# Computing the Cross Entropy loss
loss = -tf.reduce_sum(y * tf.log(output))
regularize = tf.contrib.layers.l2_regularizer(reg_scale)
params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
reg_term = sum([regularize(param) for param in params])
loss += reg_term

# Trainning with Adadelta Optimizer
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

# Calculatio of correct prediction and accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
# sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_image, train_label, test_seqlen = dataset.get_train()
test_image, test_label, test_seqlen = dataset.get_test()
len_train = len(dataset.train)
n_bins = len_train / batch_size
for epoch in xrange(training_iters):

    for i in xrange(n_bins+1):
        batch_x, batch_y, batch_seqlen = dataset.next_batch(batch_size, i+1)
        sess.run(train_step, feed_dict={rnn._inputs: batch_x, y: batch_y})

    Loss = str(sess.run(loss, feed_dict={rnn._inputs: batch_x, y: batch_x}))
    Train_accuracy = str(sess.run(accuracy, feed_dict={rnn._inputs: train_image, y: train_label}))
    Test_accuracy = str(sess.run(accuracy, feed_dict={rnn._inputs: test_image, y: test_label}))

    if epoch % 50 == 0:
        print("\rIteration: %s Loss: %s Train Accuracy: %s Test Accuracy: %s" \
              %(epoch, Loss, Train_accuracy, Test_accuracy))


