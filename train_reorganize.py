####################################################################################################################
## class oriented recoding
## problems:
## 1. Cannot easily apply dropout and L2 regularizer
## 2. The training performance seems to be stuck to a local maxima and never jumps out
####################################################################################################################

from __future__ import division
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
tf.reset_default_graph()

class convNeuralNet3d:
    def __init__(self, images, labels):
        # data
        self.images = images
        self.labels = labels
        # model
        self.convlayers()
        self.fc_layers()

    def convlayers(self):
        self.parameters = []
        images = self.images

        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 5, 5, 1, 20], dtype=tf.float32,
                                                     stddev=1e-1),name='weights')
            conv = tf.nn.conv3d(images, kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
            biases = tf.Variable(tf.constant(0.0, shape=[20], dtype=tf.float32),
                                    trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool3d(self.conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                                      padding="VALID",
                                      name='pool1')

        # conv2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([2, 3, 3, 20, 25], dtype=tf.float32,
                                                     stddev=1e-1),name='weights')
            conv = tf.nn.conv3d(self.pool1, kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
            biases = tf.Variable(tf.constant(0.0, shape=[25], dtype=tf.float32),
                                    trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool3d(self.conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                                      padding="VALID",
                                      name='pool2')

        # conv3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([2, 3, 3, 25, 30], dtype=tf.float32,
                                                     stddev=1e-1),name='weights')
            conv = tf.nn.conv3d(self.pool2, kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
            biases = tf.Variable(tf.constant(0.0, shape=[30], dtype=tf.float32),
                                    trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool3d(self.conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                                      padding="VALID",
                                      name='pool3')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            # shape = int(np.prod(self.pool3.get_shape()[1:]))
            shape = 2 * 14 * 14 * 30
            fc1w = tf.Variable(tf.truncated_normal([shape, 100], dtype=tf.float32,
                                                    stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[100], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool3_flat = tf.reshape(self.pool3, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            # self.fc1 = tf.nn.dropout(tf.nn.relu(fc1l), keep_prob=0.7) # with dropout - cannot turn off when evaluating
			# self.fc1 = tf.layers.dropout(tf.nn.relu(fc1l), rate=0.3, training=training)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([100, 2], dtype=tf.float32,
                                                    stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

    def get_accuracy(self, reg_scale=0.1, training=True):
        # reg = tf.nn.l2_loss(self.parameters[7]) + tf.nn.l2_loss(self.parameters[9])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2, labels=self.labels))
        # loss = tf.reduce_mean(loss + reg_scale * reg)
        if not training:
            correct_prediction = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            prob = tf.nn.softmax(logits=self.fc2)
            return loss, accuracy, prob
        return loss

def load_data(path):
    file = np.load(path)
    image_data = file['x']
    label_data = file['y']
    # zero-mean input
    for i in range(np.shape(image_data)[0]):
        image_data[i] = (image_data[i] - np.mean(image_data[i])) #/ np.std(image_data[i])
    return image_data, label_data

def divide_data(image_data, label_data, index=612):
    train_images = image_data[:index,]
    train_labels = label_data[:index]
    valid_images = image_data[index:,]
    valid_labels = label_data[index:]
    ## The following segmant further extracts 1 and 0 cases
    # train_data = zip(train_images, train_labels)
    # train_data0 = [data for data in train_data if data[1][1] == 0]
    # train_images0, train_labels0 = zip(*train_data0)
    # train_data1 = [data for data in train_data if data[1][1] == 1]
    # train_images1, train_labels1 = zip(*train_data1)
    # train_images0 = np.array(train_images0)
    # train_images1 = np.array(train_images1)
    # train_labels0 = np.array(train_labels0)
    # train_labels1 = np.array(train_labels1)
    # return train_images0, train_labels0, train_images1, train_labels1, valid_images, valid_labels
    return train_images, train_labels, valid_images, valid_labels

def get_batch(images, labels, batch_size):
    # Xtrain_temp, Ytrain_temp = shuffle(images, labels)
    # return Xtrain_temp[:batch_size, ], Ytrain_temp[:batch_size, ]
    return shuffle(images, labels, n_samples=batch_size)


if __name__ == '__main__':
    path = "/home/ykong24/stroke_processed_data.npz"
    # path = "/labs/colab/3DDD/stroke_processed_data.npz"
    images, labels = load_data(path)
    train_images, train_labels, valid_images, valid_labels = divide_data(images, labels)
    del images
    del labels

    learning_rate = 0.001
    reg_scale = 0.1
    batch_size = 20
    n_epoches = 50

    x = tf.placeholder(tf.float32, [None, 24, 128, 128, 1])
    y = tf.placeholder(tf.float32, [None, 2])
    cnn3d = convNeuralNet3d(x, y)
    loss = cnn3d.get_accuracy()
    accuracy = cnn3d.get_accuracy(training=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # sample_para = cnn3d.parameters[1]

    # config = tf.ConfigProto(device_count={'GPU': 0})
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for i in xrange(n_epoches):
        for j in xrange(5):
            batch_x, batch_y = get_batch(train_images, train_labels, batch_size)
            session.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # _, batch_loss, para = session.run([optimizer, loss, sample_para], feed_dict={x: batch_x, y: batch_y})
            # print i, batch_loss
            # print para

        Loss, Acc, prob = session.run(accuracy, feed_dict={x: train_images, y: train_labels})
        auc = metrics.roc_auc_score(train_labels, prob)
        print i, Loss, Acc, auc
