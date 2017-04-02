# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imread, imresize
import convert as cvt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Some constants
INPUT_FOLDER = 'sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
labels = pd.read_csv('stage1_labels.csv', index_col=0)

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

IMG_PX_SIZE = 224

def resize(image):
    image = cv2.resize(np.array(image), (IMG_PX_SIZE, IMG_PX_SIZE))
    return image

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

if __name__=='__main__':
    sample_data=[]
    # patients = patients[:5]
    for num,patient in enumerate(patients):
        try:
            label = labels.get_value(patient, 'cancer')
            if label == 1:
                label = np.array([0, 1])
            elif label == 0:
                label = np.array([1, 0])
            slices=os.listdir(INPUT_FOLDER+patient)
            new_slices = []
            for slice in slices:
                mri_file_path=INPUT_FOLDER+patient+'/'+slice
                png_file_path=INPUT_FOLDER+patient+'/temp.png'
                try:
                    # Convert the actual file
                    cvt.convert_file(mri_file_path, png_file_path)
                    img = imread(png_file_path, mode='RGB')
                    img = imresize(img, (224, 224))
                    # img=img-np.mean(img)
                    new_slices.append(img)
                    os.remove(png_file_path)
                    flag = 0
                except Exception as e:
                    print 'Cannot convert patient',num,':',e
                    os.remove(png_file_path)
                    # print('temp.png removed.')
                    flag=1
                    break

            if not flag:
                new_slices=np.array(new_slices)
                new_slices=np.float32(new_slices)
                new_slices=new_slices-np.mean(new_slices)
                sample_data.append((new_slices,label))
                print "Patient:", num, "processed."

        except KeyError as err:
            print 'Unlabeled patient:',num,"passed."

    np.save('sample_data.npy', sample_data)
    print 'pre-processed data saved.'

    # sample_data=[]
    # patients=patients[:3]
    # for num, patient in enumerate(patients):
    #     pictures=get_pixels_hu(load_scan(INPUT_FOLDER + patient))
    #     # new_pictures=[]
    #     # for picture in pictures:
    #     #     new_pictures.append(resize(picture))
    #     # pictures=resize(pictures[])  ???
    #     pictures = normalize(pictures)
    #     pictures = zero_center(pictures)
    #     sample_data.append(pictures)
    #     print("patient:",num,"processed.")



