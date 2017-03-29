import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
import glob
from skimage.io import imread
from skimage.transform import resize
import os
import subprocess
import itertools
from datetime import datetime

#from tensorflow.python.framework.ops import reset_default_graph

image_paths = glob.glob("/home/yunxiao/homework/images/*")
train = pd.read_csv('/home/yunxiao/homework/train.csv')
test = pd.read_csv('/home/yunxiao/homework/test.csv')
sample_submission = pd.read_csv('/home/yunxiao/homework/sample_submission.csv')
# try and extract and plot columns
X = train.as_matrix(columns=train.columns[2:])

margin = X[:, :64]
shape = X[:, 64:128]
texture = X[:, 128:]



def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out



class load_data():
    # data_train, data_test and le are public
    def __init__(self, train_path, test_path, image_paths, image_shape=(128, 128)):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        image_paths = image_paths
        image_shape = image_shape
        self._load(train_df, test_df, image_paths, image_shape)
        
    def _load(self, train_df, test_df, image_paths, image_shape):
        print "loading data ..."
        # load train.csv
        path_dict = self._path_to_dict(image_paths) # numerate image paths and make it a dict
        # merge image paths with data frame
        train_image_df = self._merge_image_df(train_df, path_dict)
        test_image_df = self._merge_image_df(test_df, path_dict)
        # label encoder-decoder (self. because we need it later)
        self.le = LabelEncoder().fit(train_image_df['species'])
        # labels for train
        t_train = self.le.transform(train_image_df['species'])
        # getting data
        train_data = self._make_dataset(train_image_df, image_shape, t_train)
        test_data = self._make_dataset(test_image_df, image_shape)        
        # need to reformat the train for validation split reasons in the batch_generator
        self.train = self._format_dataset(train_data, for_train=True)
        self.test = self._format_dataset(test_data, for_train=False)
        print "data loaded"
        

    def _path_to_dict(self, image_paths):
        path_dict = dict()
        for image_path in image_paths:
            num_path = int(os.path.basename(image_path[:-4]))
            path_dict[num_path] = image_path
        return path_dict

    def _merge_image_df(self, df, path_dict):
        split_path_dict = dict()
        for index, row in df.iterrows():
            split_path_dict[row['id']] = path_dict[row['id']]
        image_frame = pd.DataFrame(split_path_dict.values(), columns=['image'])
        df_image =  pd.concat([image_frame, df], axis=1)
        return df_image
    

    def _make_dataset(self, df, image_shape, t_train=None):
        if t_train is not None:
            print "loading train ..."
        else:
            print "loading test ..."
        # make dataset
        data = dict()
        # merge image with 3x64 features
        for i, dat in enumerate(df.iterrows()):
            index, row = dat
            sample = dict()
            if t_train is not None:
                features = row.drop(['id', 'species', 'image'], axis=0).values
            else:
                features = row.drop(['id', 'image'], axis=0).values
            sample['margin'] = features[:64]
            sample['shape'] = features[64:128]
            sample['texture'] = features[128:]
            if t_train is not None:
                sample['t'] = np.asarray(t_train[i], dtype='int32')
            image = imread(row['image'], as_grey=True)
            image = resize(image, output_shape=image_shape)
            image = np.expand_dims(image, axis=2)
            sample['image'] = image   
            data[row['id']] = sample
            if i % 100 == 0:
                print "\t%d of %d" % (i, len(df))
        return data

    def _format_dataset(self, df, for_train):
        # making arrays with all data in, is nessesary when doing validation split
        data = dict()
        value = df.values()[0]
        img_tot_shp = tuple([len(df)] + list(value['image'].shape))
        data['images'] = np.zeros(img_tot_shp, dtype='float32')
        feature_tot_shp = (len(df), 64)
        data['margins'] = np.zeros(feature_tot_shp, dtype='float32')
        data['shapes'] = np.zeros(feature_tot_shp, dtype='float32')
        data['textures'] = np.zeros(feature_tot_shp, dtype='float32')
        if for_train:
            data['ts'] = np.zeros((len(df),), dtype='int32')
        else:
            data['ids'] = np.zeros((len(df),), dtype='int32')
        for i, pair in enumerate(df.items()):
            key, value = pair
            data['images'][i] = value['image']
            data['margins'][i] = value['margin']
            data['shapes'][i] = value['shape']
            data['textures'][i] = value['texture']
            if for_train:
                data['ts'][i] = value['t']
            else:
                data['ids'][i] = key
        return data
    
 



TRAIN_PATH = "/home/yunxiao/homework/train.csv"
TEST_PATH = "/home/yunxiao/homework/test.csv"
IMAGE_PATHS = glob.glob("/home/yunxiao/homework/images/*.jpg")
NUM_CLASSES = 99
IMAGE_SHAPE = (128, 128, 1)
NUM_FEATURES = 64 # for all three features, margin, shape and texture
# train holds both X (input) and t (target/truth)
data = load_data(train_path=TRAIN_PATH, test_path=TEST_PATH,
                 image_paths=IMAGE_PATHS, image_shape=IMAGE_SHAPE[:2])
# to visualize the size of the dimensions of the data
print


print "@@@Shape checking of data sets@@@"
print
print "TRAIN"
print "\timages\t%s%f" % (data.train['images'].shape, data.train['images'].mean())
print "\tmargins\t%s\t%f" % (data.train['margins'].shape, data.train['margins'].mean())
print "\tshapes\t%s\t%f" % (data.train['shapes'].shape, data.train['shapes'].mean())
print "\ttextures%s\t%f" % (data.train['textures'].shape, data.train['textures'].mean())
print "\tts\t %s" % (data.train['ts'].shape)
print "\twhile training, batch_generator will onehot encode ts to (batch_size, num_classes)"
print
print "TEST"
print "\timages\t%s\t%f" % (data.test['images'].shape, data.test['images'].mean())
print "\tmargins\t%s\t%f" % (data.test['margins'].shape, data.test['margins'].mean())
print "\tshapes\t%s\t%f" % (data.test['shapes'].shape, data.test['shapes'].mean())
print "\ttextures%s\t%f" % (data.test['textures'].shape, data.test['textures'].mean())
print "\tids\t%s" % (data.test['ids'].shape)


class batch_generator():
    def __init__(self, data, batch_size=64, num_classes=99,
                 num_iterations=5e3, num_features=64, seed=42, val_size=0.1):
        print "initiating batch generator"
        self._train = data.train
        self._test = data.test
        # get image size
        value = self._train['images'][0]
        self._image_shape = list(value.shape)
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._num_iterations = num_iterations
        self._num_features = num_features
        self._seed = seed
        self._val_size = 0.1
        self._valid_split()
        print "batch generator initiated ..."

    def _valid_split(self):
        self._idcs_train, self._idcs_valid = iter(
            StratifiedShuffleSplit(self._train['ts'],
                                   n_iter=1,
                                   test_size=self._val_size,
                                   random_state=self._seed)).next()
    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def _batch_init(self, purpose):
        assert purpose in ['train', 'valid', 'test']
        batch_holder = dict()
        batch_holder['margins'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['shapes'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['textures'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['images'] = np.zeros(tuple([self._batch_size] + self._image_shape), dtype='float32')
        if (purpose == "train") or (purpose == "valid"):
            batch_holder['ts'] = np.zeros((self._batch_size, self._num_classes), dtype='float32')          
        else:
            batch_holder['ids'] = []
        return batch_holder

    def gen_valid(self):
        batch = self._batch_init(purpose='train')
        i = 0
        for idx in self._idcs_valid:
            batch['margins'][i] = self._train['margins'][idx]
            batch['shapes'][i] = self._train['shapes'][idx]
            batch['textures'][i] = self._train['textures'][idx]
            batch['images'][i] = self._train['images'][idx]
            batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='valid')
                i = 0
        if i != 0:
            yield batch, i

    def gen_test(self):
        batch = self._batch_init(purpose='test')
        i = 0
        for idx in range(len(self._test['ids'])):
            batch['margins'][i] = self._test['margins'][idx]
            batch['shapes'][i] = self._test['shapes'][idx]
            batch['textures'][i] = self._test['textures'][idx]
            batch['images'][i] = self._test['images'][idx]
            batch['ids'].append(self._test['ids'][idx])
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='test')
                i = 0
        if i != 0:
            yield batch, i
            

    def gen_train(self):
        batch = self._batch_init(purpose='train')
        iteration = 0
        i = 0
        while True:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                # extract data from dict
                batch['margins'][i] = self._train['margins'][idx]
                batch['shapes'][i] = self._train['shapes'][idx]
                batch['textures'][i] = self._train['textures'][idx]
                batch['images'][i] = self._train['images'][idx]
                batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self._batch_init(purpose='train')
                    i = 0
                    iteration += 1
                    if iteration >= self._num_iterations:
                        break

dummy_batch_gen = batch_generator(data, batch_size=64, num_classes=99, num_iterations=5e3, seed=42)
train_batch = dummy_batch_gen.gen_train().next()
valid_batch, i = dummy_batch_gen.gen_valid().next()
test_batch, i = dummy_batch_gen.gen_test().next()


print
print "@@@Shape/mean checking of batches@@@"
print
print "TRAIN"
print "\timages,", train_batch['images'].shape
print "\tmargins,", train_batch['margins'].shape
print "\tshapes,", train_batch['shapes'].shape
print "\ttextures,", train_batch['textures'].shape
print "\tts,", train_batch['ts'].shape
print
print "VALID"
print "\timages,", valid_batch['images'].shape
print "\tmargins,", valid_batch['margins'].shape
print "\tshapes,", valid_batch['shapes'].shape
print "\ttextures,", valid_batch['textures'].shape
print "\tts,", valid_batch['ts'].shape
print
print "TEST"
print "\timages,", test_batch['images'].shape
print "\tmargins,", test_batch['margins'].shape
print "\tshapes,", test_batch['shapes'].shape
print "\ttextures,", test_batch['textures'].shape
print "\tids,", len(test_batch['ids'])
# notice that mean is very different, which is why we use batch_norm in all input data in model


# contrib layers similar to wrappings used in Lasagne (for theano) or Keras
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import dynamic_rnn


# wrapping conv with batch_norm
def conv(l_in, num_outputs, kernel_size, scope, stride=1):
    return convolution2d(l_in, num_outputs=num_outputs, kernel_size=kernel_size,
                         stride=stride, normalizer_fn=batch_norm, scope=scope)

# pre-activation: http://arxiv.org/abs/1603.05027
# wrapping convolutions and batch_norm
def conv_pre(l_in, num_outputs, kernel_size, scope, stride=1):
    l_norm = batch_norm(l_in)
    l_relu = relu(l_norm)
    return convolution2d(l_relu, num_outputs=num_outputs, kernel_size=kernel_size,
                         stride=stride, activation_fn=None, scope=scope)
# easy to use pool function
def pool(l_in, scope, kernel_size=(3, 3)):
    return max_pool2d(l_in, kernel_size=kernel_size, scope=scope) # (3, 3) has shown to work better than (2, 2)


height, width, channels = IMAGE_SHAPE
# resetting the graph ...
reset_default_graph()

# Setting up placeholder, this is where your data enters the graph!
x_image_pl = tf.placeholder(tf.float32, [None, height, width, channels], name="x_image_pl")
x_margin_pl = tf.placeholder(tf.float32, [None, NUM_FEATURES], name="x_margin_pl")
x_shape_pl = tf.placeholder(tf.float32, [None, NUM_FEATURES], name="x_shape_pl")
x_texture_pl = tf.placeholder(tf.float32, [None, NUM_FEATURES], name="x_texture_pl")
is_training_pl = tf.placeholder(tf.bool, name="is_training_pl")

# Building the layers of the neural network
# we define the variable scope, so we more easily can recognise our variables later

## IMAGE
l_conv1_a = conv(x_image_pl, 16, (5, 5), scope="l_conv1_a")
l_pool1 = pool(l_conv1_a, scope="l_pool1")
l_conv2_a = conv(l_pool1, 16, (5, 5), scope="l_conv2_a")
l_pool2 = pool(l_conv2_a, scope="l_pool2")
#l_conv3_a = conv(l_pool2, 16, (5, 5), scope="l_conv3_a")
#l_pool3 = pool(l_conv3_a, scope="l_pool3")
#l_conv4_a = conv(l_pool3, 16, (5, 5), scope="l_conv4_a")
#l_pool4 = pool(l_conv3_a, scope="l_pool4")
l_flatten = flatten(l_pool2, scope="flatten")

## RNN
# define the cell of your RNN
shape_cell = tf.nn.rnn_cell.GRUCell(100)
# run the RNN as outputs, state = tf.nn.dynamic_rnn(cell, ...)
# given we run many-to-one we only care about the last state, so only
# shape_state is defined
_, shape_state = tf.nn.dynamic_rnn(cell=shape_cell,
   inputs=tf.expand_dims(batch_norm(x_shape_pl), 2), dtype=tf.float32, scope="shape_rnn")

## COMBINE
# use margin, shape and texture only
#features = tf.concat(concat_dim=1, values=[x_margin_pl, x_shape_pl, x_texture_pl], name="features")
# uncomment to use image only
#features = l_flatten
# uncomment to use margin, rnn_state on shape and texture only
features = tf.concat(concat_dim=1, values=[l_flatten,x_margin_pl, shape_state, x_texture_pl], name="features")
features = batch_norm(features, scope='features_bn')
l2 = fully_connected(features, num_outputs=256, activation_fn=relu,
                     normalizer_fn=batch_norm, scope="l2")
l2 = dropout(l2, is_training=is_training_pl, scope="l2_dropout")
y = fully_connected(features, NUM_CLASSES, activation_fn=softmax, scope="y")



# PRINT NETWORK (good practice to also include outcommented code when using it)

print "x_image_pl,", x_image_pl.get_shape()
print "x_margin_pl,", x_margin_pl.get_shape()
print "x_shape_pl,", x_shape_pl.get_shape()
print "x_texture_pl,", x_texture_pl.get_shape()
print "features,", features.get_shape()
print "y,", y.get_shape()

# for the MLP
#print "l2,", l2.get_shape()
# for the RNN
#print "shape_state,", shape_state.get_shape()
# for the CNN
#print "l_conv1_a,", l_conv1_a.get_shape()
#...
#print "l_pool4,", l_pool4.get_shape()
#print "l_flatten,", l_flatten.get_shape()

clip_norm = 1
# y_ is a placeholder variable taking on the value of the target batch.
ts_pl = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="targets_pl")
lr_pl = tf.placeholder(tf.float32, [], name="learning_rate_pl")

def loss_and_acc(preds):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(ts_pl * tf.log(preds+1e-10), reduction_indices=[1])
    # averaging over samples
    loss = tf.reduce_mean(cross_entropy)
    # if you want regularization
    #reg_scale = 0.0001
    #regularize = tf.contrib.layers.l2_regularizer(reg_scale)
    #params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #reg_term = sum([regularize(param) for param in params])
    #loss += reg_term
    # calculate accuracy
    argmax_y = tf.to_int32(tf.argmax(preds, dimension=1))
    argmax_t = tf.to_int32(tf.argmax(ts_pl, dimension=1))
    correct = tf.to_float(tf.equal(argmax_y, argmax_t))
    accuracy = tf.reduce_mean(correct)
    return loss, accuracy, argmax_y

# loss, accuracy and prediction
loss, accuracy, prediction = loss_and_acc(y)

loss_valid = loss
accuracy_valid = accuracy
loss_valid, accuracy_valid, _ = loss_and_acc(y)

# defining our optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

# applying the gradients
grads_and_vars = optimizer.compute_gradients(loss)
gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
clipped_gradients, global_norm = (
    tf.clip_by_global_norm(gradients, clip_norm) )
clipped_grads_and_vars = zip(clipped_gradients, variables)

# make training op for applying the gradients
train_op = optimizer.apply_gradients(clipped_grads_and_vars)

# make tensorboard summeries
#tf.scalar.summary('train/global gradient norm', global_norm)
#tf.scalar.summary('train/loss', loss)
#tf.scalar.summary('train/accuracy', accuracy)
#tf.scalar.summary('validation/loss', loss_valid)
#tf.scalar.summary('validation/accuracy', accuracy_valid)

#Test the forward pass
_img_shape = tuple([45]+list(IMAGE_SHAPE))
_feature_shape = (45, NUM_FEATURES)
_x_image = np.random.normal(0, 1, _img_shape).astype('float32') #dummy data
_x_margin = np.random.normal(0, 1, _feature_shape).astype('float32')
_x_shape = np.random.normal(0, 1, _feature_shape).astype('float32')
_x_texture = np.random.normal(0, 1, _feature_shape).astype('float32')

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# initialize the Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
# test the forward pass
sess.run(tf.initialize_all_variables())
feed_dict = {x_image_pl: _x_image,
             x_margin_pl: _x_margin,
             x_shape_pl: _x_shape,
             x_texture_pl: _x_texture,
             is_training_pl: False}
res_forward_pass = sess.run(fetches=[y], feed_dict=feed_dict)
print "y", res_forward_pass[0].shape

#Training Loop
BATCH_SIZE = 64
ITERATIONS = 1e4
LOG_FREQ = 10
VALIDATION_SIZE = 0.1 # 0.1 is ~ 100 samples for valition
SEED = 42
DROPOUT = False
LEARNING_RATE = 0.0005
VALID_EVERY = 100

batch_gen = batch_generator(data, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES,
                            num_iterations=ITERATIONS, seed=SEED, val_size=VALIDATION_SIZE)

# setup and write summaries
#timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#summaries_path = "tensorboard/%s/logs" % (timestamp)
#summaries = tf.merge_all_summaries()
#summarywriter = tf.train.SummaryWriter(summaries_path, sess.graph)

train_loss = []
train_acc = []
print "\ttrain_loss \ttrain_acc \tvalid_loss \tvalid_acc"
for i, batch_train in enumerate(batch_gen.gen_train()):
    if i>=ITERATIONS:
        break
    #fetches_train = [train_op, loss, accuracy, summaries]
    fetches_train = [train_op, loss, accuracy]
    feed_dict_train = {
        x_image_pl: batch_train['images'],
        x_margin_pl: batch_train['margins'],
        x_shape_pl: batch_train['shapes'],
        x_texture_pl: batch_train['textures'],
        ts_pl: batch_train['ts'],
        is_training_pl: DROPOUT,
        lr_pl: LEARNING_RATE,

    }
    res_train = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)
    #if i % LOG_FREQ == 0:
    #    summarywriter.add_summary(res_train[3], i)
    train_loss.append(res_train[1])
    train_acc.append(res_train[2])

    # validate
    if i % VALID_EVERY == 0:
        cur_acc = 0
        cur_loss = 0
        tot_num = 0
        # batch validation
        for batch_valid, num in batch_gen.gen_valid():
            # fetches and feed_dict for validation
            #fetches_valid = [loss_valid, accuracy_valid, summaries]
            fetches_valid = [loss_valid, accuracy_valid]
            feed_dict_valid = {
                x_image_pl: batch_valid['images'],
                x_margin_pl: batch_valid['margins'],
                x_shape_pl: batch_valid['shapes'],
                x_texture_pl: batch_valid['textures'],
                ts_pl: batch_valid['ts'],
                is_training_pl: False,
            }
            # run validation
            res_valid = sess.run(fetches=fetches_valid, feed_dict=feed_dict_valid)
            # tensorboard and costs
            #summarywriter.add_summary(res_valid[2], i)
            cur_loss += res_valid[0]*num
            cur_acc += res_valid[1]*num
            tot_num += num
        valid_loss = cur_loss / float(tot_num)
        valid_acc = (cur_acc / float(tot_num)) * 100
        train_loss = sum(train_loss) / float(len(train_loss))
        train_acc = sum(train_acc) / float(len(train_acc)) * 100
        print "%d:\t  %.2f\t\t  %.1f\t\t  %.2f\t\t  %.1f" % (i, train_loss, train_acc, valid_loss, valid_acc)
        train_loss = []
        train_acc = []



