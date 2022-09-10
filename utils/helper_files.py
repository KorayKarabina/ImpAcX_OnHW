from utils import config
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
from tsai.all import *
from platform import system
from utils import folders_and_files
import pathlib




def OnHW_create_paired_datasets(labels,data,multiclass_pairs,case,dependency,k_fold_number,name):
    datasets = {}
    for i, pair in enumerate(multiclass_pairs):

        path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.DL_MODELS_AND_DATA)
        folder_name = f"{case}_{dependency}_{k_fold_number}_mc"
        path_to_model_data = path_to_windows(os.path.join(path_to_models_and_data, folder_name, f"x{name}_{pair[0]}_{pair[1]}.npy"))
        path_to_model_label = path_to_windows(os.path.join(path_to_models_and_data, folder_name, f"y{name}_{pair[0]}_{pair[1]}.npy"))
        folders_and_files.make_folder_at(os.path.join("OnHW_models",path_to_models_and_data), folder_name)
        if os.path.exists(path_to_model_data) and os.path.exists(path_to_model_label):
            data_temp = np.load(path_to_model_data, allow_pickle=True)
            labels_temp = np.load(path_to_model_label, allow_pickle=True)
        else:
            a=0
            data_temp = data.copy()
            labels_temp = labels.copy()
            mask = [(data in pair) for data in labels_temp]
            data_temp, labels_temp = data_temp[mask], labels_temp[mask]
            np.save(path_to_model_data, data_temp)
            np.save(path_to_model_label,labels_temp)
        
        #This is back outside the if else
        datasets[pair] = (labels_temp,data_temp)
    return datasets


def OnHW_ML_read_filtered_data(case, dependency, k_fold_number):
    path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
    folder_name = f"{case}_{dependency}_{k_fold_number}"

    train_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_X_filtered.npy"), allow_pickle=True)
    train_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_y_filtered.npy"), allow_pickle=True)
    test_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_X_filtered.npy"), allow_pickle=True)
    test_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_y_filtered.npy"), allow_pickle=True)

    return train_X_filtered, train_y_filtered, test_X_filtered, test_y_filtered


def tsai_ready_data(case, dependency, k_fold_number):
    train_X_filtered, train_y_filtered, test_X_filtered, test_y_filtered = OnHW_ML_read_filtered_data(case, dependency, k_fold_number)

    trainX = X_to_fixlength(train_X_filtered)
    testX = X_to_fixlength(test_X_filtered)
    trainy = train_y_filtered.copy()
    testy = test_y_filtered.copy()

    # reshaping (num_samples, num_timesteps, num_features) --> (num_samples, num_features, num_timesteps)
    trainX = trainX.swapaxes(1, 2).astype(float)
    testX = testX.swapaxes(1, 2).astype(float)

    return trainX, trainy, testX, testy

def X_to_fixlength(X, maxlen=config.MAXLEN):
    print((X.shape,maxlen))
    return pad_sequences(X, maxlen=maxlen, truncating='post')

def OnHW_DL_load_learn_by_name(case, dependency, k_fold_number, name):
    
    folder_name = f"{case}_{dependency}_{k_fold_number}"
    model_name = f"model_{name}.pkl"
    path_to_model = os.path.join(config.BASE_OUTPUT, config.DL_MODELS_AND_DATA, folder_name, model_name)
    try:
        plt = platform.system()
        if plt == 'Windows': path_to_model = path_to_model.replace("'","_").replace(":","")
        learn = load_learner(path_to_model)
    except:
        learn = load_learner(path_to_model)

    return learn

def path_to_windows(path):
    # if (system() == 'Windows'):
    #     path = os.path.join("OnHW_models",path)
    #     path = path.replace("{","_").replace("}","_").replace(":","_").replace("'","")
    return path
