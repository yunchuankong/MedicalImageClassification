import tensorflow as tf
import numpy as np
import os
import pandas as pd
from vgg_transfer_utils import load_scan,get_pixels_hu,resample,normalize,triplicate,resizeImage,vgg16
# from scipy.misc import imresize

def process(patientID, label, rsp=False):
    patient = load_scan(patientID)
    slices = get_pixels_hu(patient)
    if rsp:
        slices,spacing = resample(slices, patient, [1,1,1])
    slices = resizeImage(triplicate(normalize(slices)),224,224)

    # for slice in slices:
    #     slice = triplicate(normalize(imresize(slice, (224, 224))))
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    bottle_neck = sess.run(vgg.bn, feed_dict={vgg.imgs: slices})
    slices=None
    vgg=None
    imgs=None

    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])

    return bottle_neck, label



if __name__ == "__main__":
    os.chdir('../../labs/colab/3DDD/kaggle_data')
    INPUT_FOLDER = 'sample_images/'
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    labels = pd.read_csv('stage1_labels.csv', index_col=0)
    sample_data=[]
    patients=patients[:3]
    for num, patientID in enumerate(patients):
        try:
            label = labels.get_value(patientID, 'cancer')
            rnnInput,label = process(INPUT_FOLDER + patientID, label, rsp=False)
            sample_data.append( (rnnInput,label) )
            print 'Patient',num,'processed.'

        except KeyError as err:
            print 'Unlabeled patient:', num, "passed."

    np.save('sample_data.npy', sample_data)
    print 'pre-processed data saved.'

