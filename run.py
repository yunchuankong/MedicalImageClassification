import os
import pandas as pd
import numpy as np
from data_process import process
INPUT_FOLDER = '/labs/colab/3DDD/kaggle_data/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
labels = pd.read_csv('/labs/colab/3DDD/kaggle_data/stage1_labels.csv', index_col=0)
sample_data = []

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


