from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
reset_default_graph()


# Parameters
learning_rate = 0.01
training_iters = 3000000
batch_size = 80
display_step = 500
n_embedding = 200
n_hidden = 100 # hidden layer num of features
n_classes = 2 # linear sequence or not
n_features = 4096
penalty_scale = 0.0005
# Network Parameters


def data_summary(data):
    summary = dict()
    summary["len"] = len(data)
    summary["n_slices"] = np.array(list(map(lambda x: len(x[0]), data)))
    summary["n_features"] = n_features
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
    def __init__(self, data, summary, test_prop = 0.1):
        self.images = []
        self.labels = []
        self.seqlen = summary["n_slices"]
        max_seq_len = summary["max_seq_len"]
        for i in range(summary["len"]):
            self.images.append(np.concatenate((data[i][0],np.zeros([max_seq_len-self.seqlen[i],n_features]))))
            #pad each sequence to reach 'max_seq_len'
            self.labels.append(data[i][1])
        #Obtain the training set and test set for RNN
        self.test = np.random.choice(summary["len"],round(test_prop*summary["len"]), replace = False)
        self.train =  np.setdiff1d(np.array(range(summary["len"])), self.test)


    def next_train(self, batch_size):
        """ Return a batch of data for training.
        """
        len_train = len(self.train)
        batch_id = np.random.choice(len_train, batch_size, replace=False)
        batch_images = []
        batch_labels = []
        batch_seqlen = []
        for i in batch_id:
            idx = self.train[i]
            batch_images.append(self.images[idx])
            batch_labels.append(self.labels[idx])
            batch_seqlen.append(self.seqlen[idx])
            
        return batch_images, batch_labels, batch_seqlen
    
    
    def get_test(self):
        """Return the selected test dataset.
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

bottle_neck_data = np.load("/labs/colab/3DDD/kaggle_data/kaggle_processed_data/resampled_transfer_learning_data_new.npy")
bottle_neck_summary = data_summary(bottle_neck_data)
dataset = DataBatchGenerator(bottle_neck_data,bottle_neck_summary,0.10)
seq_max_len = bottle_neck_summary["max_seq_len"]

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, n_features])
y = tf.placeholder("float", [None, n_classes])

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'in': tf.Variable(tf.random_normal([n_features, n_embedding])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'in':tf.Variable(tf.random_normal([n_embedding])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):
    #use LSTM
    x = tf.unstack(x, seq_max_len, 1)
    x_embedding = list(map(lambda ts: tf.matmul(ts, weights['in']) + biases['in'], x ))
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x_embedding, dtype=tf.float32,
                                sequence_length=seqlen)
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']


#def dynamicRNN_attention(x, seqlen, weights, biases):
    


pred = dynamicRNN(x, seqlen, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=penalty_scale, scope=None)
all_variable = tf.trainable_variables() # all vars of your graph
cost += tf.contrib.layers.apply_regularization(l1_regularizer, all_variable) #use L1 penalty

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
#with tf.Session(config = tf.ConfigProto(device_count = {'GPU':0})) as sess:
with tf.Session() as sess:
    sess.run(init)
    step = 1
    test_image, test_label, test_seqlen = dataset.get_test()
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = dataset.next_train(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate test accuracy
            test_acc = sess.run(accuracy, feed_dict={x: test_image, y: test_label,
                                      seqlen: test_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", batch_loss= " + \
                  "{:.6f}".format(loss) + ", train_acc= " + \
                  "{:.5f}".format(train_acc) +", test_acc= " +\
                 "{:.5f}".format(test_acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_image, y: test_label,
                                      seqlen: test_seqlen}))