from unicodedata import bidirectional
from utils import config
from utils import folders_and_files

import os
import sys
import numpy as np

from tsai.all import *
import optuna
from optuna.integration import FastAIPruningCallback
from tensorflow.keras.preprocessing.sequence import pad_sequences


def X_to_fixlength(X, maxlen=config.MAXLEN):
    return pad_sequences(X, maxlen=maxlen, truncating='post')

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



def objective(trial: optuna.Trial):

    model_name, case, dependency, k_fold_number = config.TO_OPTIMIZE[0], config.TO_OPTIMIZE[1], config.TO_OPTIMIZE[2], config.TO_OPTIMIZE[3]
    trainX, trainy, testX, testy = tsai_ready_data(case, dependency, k_fold_number)
    X, y, splits = combine_split_data([trainX, testX], [trainy, testy])
    
    # search through all float values between 0.0 and 0.5 in log increment steps
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-2,
        log=True)  

    # search through epochs between 25 and 100 in increments of 25
    epoch_number = trial.suggest_int(
        "epoch_number", 25, 100, step=25)  

    if model_name =='InceptionTimePlus':
        # InceptionPlus parameters
        nf = trial.suggest_categorical('num_filters', [4, 8, 16, 32, 40, 48, 56, 64, 128])
        depth = trial.suggest_int('depth', 1, 15, step=1)
        fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.9,step=.1)
    
        arch = InceptionTimePlus
        arch_config = {'nf': nf, 
                    'fc_dropout': fc_dropout,
                    'depth': depth}
    
    elif model_name =='LSTM':
        # LSTM parameters
        n_layers = trial.suggest_int('n_layers', 1, 5, step=1)
        fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.9,step=.1)
        rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.9,step=.1)
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])        

        arch = LSTM
        arch_config = {'n_layers': n_layers, 
                        'fc_dropout': fc_dropout,  
                        'rnn_dropout': rnn_dropout,  
                        'bidirectional': bidirectional}

    elif model_name =='LSTM_FCN':
        # LSTM_FCN parameters
        rnn_layers = trial.suggest_int('n_layers', 1, 5, step=1)
        fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.9,step=.1)
        rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.9,step=.1)
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])        

        arch = LSTM_FCN
        arch_config = {'rnn_layers': rnn_layers, 
                        'fc_dropout': fc_dropout,  
                        'rnn_dropout': rnn_dropout,  
                        'bidirectional': bidirectional}

    elif model_name =='MLSTM_FCN':
        # MLSTM_FCN parameters
        rnn_layers = trial.suggest_int('n_layers', 1, 5, step=1)
        fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.9,step=.1)
        rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.9,step=.1)
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])        

        arch = MLSTM_FCN
        arch_config = {'rnn_layers': rnn_layers, 
                        'fc_dropout': fc_dropout,  
                        'rnn_dropout': rnn_dropout,  
                        'bidirectional': bidirectional}


    batch_tfms = TSStandardize(by_sample=True)
    learn = TSClassifier(
        X, y, splits=splits, bs=[64, 128], batch_tfms=batch_tfms,
        arch=arch, arch_config=arch_config,
        metrics=accuracy, cbs=FastAIPruningCallback(trial))


    with ContextManagers(
            [learn.no_logging(),
             learn.no_bar()]):  # [Optional] this prevents fastai from printing anything during training
        learn.fit_one_cycle(epoch_number, lr_max=learning_rate)

    # Return the objective value
    return learn.recorder.values[-1][1]  # return the validation loss value of the last epoch


'''Currently, the parameter "TO_OPTIMIZE" in the configuration file (config.py) is set to
TO_OPTIMIZE = ['InceptionTimePlus', 'lower', 'indep', 0]
Change "TO_OPTIMIZE" in the configuration file to optimize other models over other datasets
The model and optimal parameters info can be copied to 
"DL_OPT_MODEL_LIST" in "OnHW_ML_DL_baseline_and_optimized_train_and_eval.py" for training and evaluation'''
if __name__ == "__main__":

    model_name, case, dependency, k_fold_number = config.TO_OPTIMIZE[0], config.TO_OPTIMIZE[1], config.TO_OPTIMIZE[2], config.TO_OPTIMIZE[3]
    path_to_results = os.path.join(config.BASE_OUTPUT, config.OPTUNA, f'opt_{model_name}_{case}_{dependency}_{k_fold_number}.txt')
    with open(path_to_results, 'w') as f:
        
        sys.stdout = f
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))