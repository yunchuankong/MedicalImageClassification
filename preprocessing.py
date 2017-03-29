# https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet/notebook
import numpy as np
import pandas as pd
import dicom
import os
# import matplotlib.pyplot as plt
# import cv2
# import math

# IMG_SIZE_PX = 50
# SLICE_COUNT = 20

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(a):
    return sum(a) / len(a)


def process_data(patient,labels_df
                   # ,img_px_size=50, hm_slices=20, 
                   # ,visualize=False
                   ):
    
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    # new_slices = []
    # slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
    
    # chunk_sizes = math.ceil(len(slices) / hm_slices)
    # for slice_chunk in chunks(slices, chunk_sizes):
        # slice_chunk = list(map(mean, zip(*slice_chunk)))
        # new_slices.append(slice_chunk)

    # if len(new_slices) == hm_slices-1:
        # new_slices.append(new_slices[-1])

    # if len(new_slices) == hm_slices-2:
        # new_slices.append(new_slices[-1])
        # new_slices.append(new_slices[-1])

    # if len(new_slices) == hm_slices+2:
        # new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        # del new_slices[hm_slices]
        # new_slices[hm_slices-1] = new_val
        
    # if len(new_slices) == hm_slices+1:
        # new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        # del new_slices[hm_slices]
        # new_slices[hm_slices-1] = new_val

    # if visualize:
        # fig = plt.figure()
        # for num,each_slice in enumerate(slices):
            # y = fig.add_subplot(4,5,num+1)
            # y.imshow(each_slice, cmap='gray')
        # plt.savefig(patient)

    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])
        
    return np.array(slices),label

#                                               stage 1 for real.
data_dir = 'sample_images/'
patients = os.listdir(data_dir)
labels = pd.read_csv('stage1_labels.csv', index_col=0)

processed_data = []
for num,patient in enumerate(patients):
    # if num % 100 == 0:
        # print(num)
    print(num+1)
    try:
        img_data,label = process_data(patient,labels) # ,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT
        #print(img_data.shape,label)
        processed_data.append([img_data,label])
    except KeyError as e:
        print('This is unlabeled data!')

np.save('processed_data-{}.npy'.format(len(patients)), processed_data)