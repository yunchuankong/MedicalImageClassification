import numpy as np

def divide(dataset,train_prop = 0.9):
    indicator = list(map(lambda x: x[1][1], dataset))
    ind_case = [i for i, x in enumerate(indicator) if x == 1]
    ind_control = [i for i, x in enumerate(indicator) if x == 0]
    n_control = len(ind_control)
    n_case = len(ind_case)
    div_control = int(round(n_control * train_prop))
    div_case = int(round(n_case * train_prop))
    print "n_control: ",n_control, ",n_case: ",n_case, \
        ",train_control: ", div_control, ",train_case: ",div_case
    ind_train =  ind_control[:div_control] + ind_case[:div_case]
    ind_test =  ind_control[div_control:] + ind_case[div_case:]
    data = dict()
    data['train'] = np.array([dataset[i] for i in ind_train])
    data['test'] = np.array([dataset[i] for i in ind_test])
    return data

if __name__ == "__main__":
    dat = np.load(\
        "/labs/colab/3DDD/kaggle_data/kaggle_processed_data/resampled_transfer_learning_data_new.npy")
    processed_dat = divide(dat)
    np.save(\
        "/labs/colab/3DDD/kaggle_data/kaggle_processed_data/resampled_data_new_divided.npy",processed_dat)


## When loading, use the following:
# dat = np.load("/labs/colab/3DDD/kaggle_data/kaggle_processed_data/resampled_data_new_divided.npy")
# dat = dat.item()
# np.shape(dat['train']) # (1257, 2)
# np.shape(dat['test']) # (139, 2)
