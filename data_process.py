import tensorflow as tf
import numpy as np
import os
import pandas as pd
from vgg_transfer_utils import load_scan,get_pixels_hu,resample,normalize,triplicate,_3D_images_resize,vgg16
from tensorflow.python.framework.ops import reset_default_graph
# from scipy.misc import imresize

def process(patientID, label, rsp=False):
    patient = load_scan(patientID)
    slices = get_pixels_hu(patient)
    if rsp:
        slices,spacing = resample(slices, patient, [1,1,1])
    tri_slices = _3D_images_resize(triplicate(normalize(slices)),224,224)

    # for slice in slices:
    #     slice = triplicate(normalize(imresize(slice, (224, 224))))
    reset_default_graph()
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, '/home/yunxiao/VGG/vgg16_weights.npz', sess)
    bottle_neck = sess.run(vgg.bn, feed_dict={vgg.imgs: tri_slices})
    del slices, tri_slices, sess, vgg,imgs
    

    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])

    return bottle_neck, label
