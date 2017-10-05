from __future__ import division
import tensorflow as tf
import numpy as np
import os
import csv
from sklearn.utils import shuffle

n_depth = 24
n_input_x = 128
n_input_y = 128
n_classes = 2

pathDicom = os.listdir("/labs/colab/3DDD/stroke_data_update")
pathDicom3 = []
item_values = []

for item in pathDicom:
    substring5 = "stroke"


    if (substring5 in item):

        continue

    else:

        item2 = '/labs/colab/3DDD/stroke_data_update/' + item
        item_values.append(item)
        pathDicom2 = os.listdir(item2)
        substring1 = 'ADC'
        substring2 = 'Apparent_Diffusion_Coefficient_(mm2s)'
        gg = 0
        for i in range(len(pathDicom2)):
            if substring1 in pathDicom2[i]:
                item3 = item2 + '/' + pathDicom2[i]
                gg += 1
                break

            if substring2 in pathDicom2[i]:
                item3 = item2 + '/' + pathDicom2[i]
                gg += 1
                break

        pathDicom3.append(item3)

All_Files = []
All_labels = []

with open('/labs/colab/3DDD/stroke_data_update/stroke_data_augmented_outcomes_update.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        All_Files.append(row[0])
        All_labels.append(row[2])



Input_data = []
label = []
count = -1


counter = 0
for item in pathDicom3:
    count = count + 1
    try:
        item_index = All_Files.index(item_values[count])
        label.append(All_labels[item_index])
    except:
        f = 0

n_examples = len(label)



FeedData = []
Files = []
P_Files = []
for num in range (len(pathDicom3)):
        for i in range (n_depth):

                f = open(pathDicom3[num] + "/IM-"+str(i+1)+".txt")
                for row in f.readlines():
                    for word in row.split(','):
                       if word != '0\n':
                           if word == '0':
                                Files.append(int(word))
                           else:
                               Files.append(float(word))
                       else:

                           Files.append(0)

                Files_Array = np.array(Files)
                Files_Array = np.float32(Files_Array)

                Files = []
                Files_Image = np.reshape(Files_Array, (n_input_x, n_input_y))
                P_Files.append(Files_Image)

        FeedData.append(P_Files)
        P_Files = []

FeedData1 = np.array(FeedData)

del FeedData

#normalization
size = FeedData1.shape
print size

MAX_Dataset = np.amax(FeedData1)
Normalized_FeedData1 = FeedData1/MAX_Dataset

Data_Input = FeedData1.reshape((n_examples, n_depth, n_input_x, n_input_y, 1)) ## the final "1" is the channel

del FeedData1

label_data = np.zeros([n_examples, 2])

print(len(label))

for i in range(n_examples):
    if label[i] == '1':
        label_data[i, 1] = 1
    else:
        label_data[i, 0] = 1
		
np.savez('/labs/colab/3DDD/stroke_processed_data.npz', x=Data_Input, y=label_data)

#######################################################################################################################
# file = np.load("stroke_processed_data.npz")
# Data_Input = file['x']
# label_data = file['y']

# train_images = image_data[124:,]
# train_labels = label_data[124:]
#
# train_data = zip(train_images,train_labels)
# train_data0 = [data for data in train_data if data[1][1]==0]
# train_images0, train_labels0 = zip(*train_data0)
# train_data1 = [data for data in train_data if data[1][1]==1]
# train_images1, train_labels1 = zip(*train_data1)
# train_images0 = np.array(train_images0)
# train_images1 = np.array(train_images1)
# train_labels0 = np.array(train_labels0)
# train_labels1 = np.array(train_labels1)
# del train_data
# del train_data0
# del train_data1




