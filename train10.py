####################################################################################################################
### Replicating Ali's cnn settings in the final report
####################################################################################################################

from __future__ import division
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
tf.reset_default_graph()
resume = False

# load data
# file = np.load("/labs/colab/3DDD/stroke_processed_data.npz")
file = np.load("/home/ykong24/stroke_processed_data.npz")
Data_Input = file['x']
label_data = file['y']

Data_Input_shape = np.shape(Data_Input)
n_depth = Data_Input_shape[1]
n_input_x = Data_Input_shape[2]
n_input_y = Data_Input_shape[3]
n_classes = 2

learning_rate = 0.0001
reg_scale = 0.1
max_iter = 200
batch_sz = 20
N_Conv_layer1 = 20
N_Conv_layer2 = 25
N_Conv_layer3 = 30
N_W_fc1 = N_b_fc1 = 100

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def cnl3d(input, w, b):
    output = tf.nn.conv3d(input, w, strides=[1, 1, 1, 1, 1], padding="VALID")
    output = tf.nn.bias_add(output, b)
    output = tf.nn.relu(output)
    return tf.nn.max_pool3d(output, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="VALID")


x = tf.placeholder(tf.float32, [None, n_depth, n_input_x, n_input_y, 1]) ## [24,128,128]
y_ = tf.placeholder(tf.float32, [None, n_classes])

W_cv1 = weight_variable([3, 5, 5, 1, N_Conv_layer1])
b_cv1 = bias_variable([N_Conv_layer1])
conv1 = cnl3d(x, W_cv1, b_cv1)

W_cv2 = weight_variable([2, 3, 3, N_Conv_layer1, N_Conv_layer2]) # change the dimension of filters to odd numbers
b_cv2 = bias_variable([N_Conv_layer2])
conv2 = cnl3d(conv1, W_cv2, b_cv2)

W_cv3 = weight_variable([2, 3, 3, N_Conv_layer2, N_Conv_layer3])
b_cv3 = bias_variable([N_Conv_layer3])
conv3 = cnl3d(conv2, W_cv3, b_cv3)

# MLP
W_fc1 = weight_variable([2 * 14 * 14 * N_Conv_layer3, N_W_fc1])
b_fc1 = bias_variable([N_b_fc1])

h_pool2_flat = tf.reshape(conv3, [-1, 2 * 14 * 14 * N_Conv_layer3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([N_W_fc1, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# define loss & optimizer
reg = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
loss = tf.reduce_mean(loss + reg_scale * reg) # with L2 regularizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# var_grad = tf.gradients(cross_entropy, [W_cv1,W_cv2,W_cv3,W_fc1,W_fc2])[0]
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_score = tf.nn.softmax(logits=y_conv)

# train
# Y_softmax = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2) ## ??? for AUC construction?

## (125 + 28 + 31 ) * 4 = 184 * 4;  t + v = 612 + 124
Training_set = Data_Input[124:,]
training_labels = label_data[124:]
# Training_set, training_labels = shuffle(Training_set, training_labels)

Valid_set = Data_Input[:124,]
Valid_label = label_data[:124]

del Data_Input
del label_data

init = tf.global_variables_initializer()
saver=tf.train.Saver(max_to_keep=100)
n_batches = int(len(Training_set)/batch_sz)
print n_batches

Loss = np.zeros(len(range(max_iter))).astype(float)
training_accuracy = np.zeros(len(range(max_iter))).astype(float)
training_auc = np.zeros(len(range(max_iter))).astype(float)
valid_accuracy = np.zeros(int(len(range(max_iter))/10)).astype(float)
valid_auc = np.zeros(int(len(range(max_iter))/10)).astype(float)

y_score_train = y_score_valid = None

# config = tf.ConfigProto(device_count={'GPU': 0})
# sess = tf.Session(config=config)
with tf.Session() as session:
    if resume:
        new_saver = tf.train.import_meta_graph('199model.ckpt.meta')
        new_saver.restore(session, tf.train.latest_checkpoint('./'))
    else:
        session.run(init)

    with tf.device('/gpu:0'):
        k = 0
        for i in xrange(0,max_iter):
            Xtrain_temp, Ytrain_temp = shuffle(Training_set, training_labels)
            shuf = shuffle(range(n_batches))
            n_batches_half = int(n_batches/4)
            for j in xrange(n_batches_half):

                batch1 = Xtrain_temp[shuf[j] * batch_sz:(shuf[j] * batch_sz + batch_sz), ]
                batch2 = Ytrain_temp[shuf[j] * batch_sz:(shuf[j] * batch_sz + batch_sz), ]


                # if j % 5 == 0:
                #     train_accuracy = accuracy.eval(feed_dict={x: batch1, y_: batch2,  keep_prob: 1})
                #     print("Step %d, BatchNumber: %d, training accuracy %g" % (i, j, train_accuracy))

                optimizer.run(feed_dict={x: batch1, y_: batch2,  keep_prob: 0.95})
                # del grad
                del batch1
                del batch2

            del Xtrain_temp
            del Ytrain_temp
            # training_accuracy[i] = accuracy.eval(feed_dict={x: Training_set, y_: training_labels, keep_prob: 1})
            training_accuracy[i], y_score_train, Loss[i] = session.run([accuracy, y_score, loss],
                                                              feed_dict={x: Training_set, y_: training_labels,
                                                                         keep_prob: 1})
            training_auc[i] = metrics.roc_auc_score(training_labels, y_score_train)
            print ("Step: %d, Training Accuracy: %g" % (i,training_accuracy[i]))
            print ("Step: %d, Training AUC: %g" % (i, training_auc[i]))
            print ("Step: %d, Training Loss: %g" % (i, Loss[i]))

            if i % 10 == 9:
                # valid_accuracy[k] = accuracy.eval(feed_dict={x: Valid_set, y_: Valid_label, keep_prob: 1})
                valid_accuracy[k], y_score_valid = session.run([accuracy, y_score],
                                                                  feed_dict={x: Valid_set, y_: Valid_label,
                                                                             keep_prob: 1})
                valid_auc[k] = metrics.roc_auc_score(Valid_label, y_score_valid)
                print ("\n**** Step: %d, Val Accuracy: %g ****"  % (i,valid_accuracy[k]))
                print ("**** Step: %d, Val AUC: %g ****\n" % (i, valid_auc[k]))
                k += 1

            if i > 100 and i % 100 == 99:
                save_path = saver.save(session, "model"+str(i))


        # Test_Accuracy = accuracy.eval(feed_dict={x: Test_set, y_: Test_label, keep_prob: 1})
        # print("Test Accuracy: %g" % (Test_Accuracy))
    np.savetxt("Loss10.csv", Loss, delimiter=",")
    np.savetxt("valid_accuracy10.csv", valid_accuracy, delimiter=",")
    np.savetxt("valid_auc10.csv", valid_auc, delimiter=",")
    np.savetxt("train_accuracy10.csv", training_accuracy, delimiter=",")
    np.savetxt("train_auc10.csv", training_auc, delimiter=",")




