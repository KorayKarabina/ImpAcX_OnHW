from pickle import TRUE
from string import ascii_letters, ascii_lowercase, ascii_uppercase
from utils import config
from utils import folders_and_files
from platform import system

import os
import pandas as pd


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tsai.all import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys

import utils.helper_files as hf

warnings.filterwarnings('ignore')
ML_MODEL_AND_SCALER_DICT = {'SVM_rbf': [SVC(kernel='rbf', decision_function_shape='ovo', random_state=config.RANDOM_STATE), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'SVM_linear': [SVC(kernel='linear', decision_function_shape='ovo', random_state=config.RANDOM_STATE), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'RFC': [RandomForestClassifier(criterion='gini', n_estimators=100, random_state=config.RANDOM_STATE), None],
                         'DT': [DecisionTreeClassifier(criterion='gini', random_state=config.RANDOM_STATE), None],
                         'ET': [ExtraTreesClassifier(criterion='gini', n_estimators=100, random_state=config.RANDOM_STATE), None],
                         'kNN': [KNeighborsClassifier(), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'LogReg': [LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         }

# A list of DL baseline models
DL_MODEL_LIST = [(LSTM_FCN, {}),
                 (LSTM_FCN, {'bidirectional': True}),
                 (LSTM, {'n_layers':2, 'bidirectional': False}),
                 (LSTM, {'n_layers':2, 'bidirectional': True}),
                 (MLSTM_FCN, {}),
                 (MLSTM_FCN, {'bidirectional': True}),
                 (FCN, {}),
                 (ResNet, {}),
                 (ResCNN, {}),
                 (InceptionTime, {}),
                 (XceptionTime, {}),
                 (XCM, {})]

# A list of optimized dl models
# Optimizations are based on experiments ovet three cases: lower, upper, both
# In each of these 3 cases, experiments have been performed over the fold-0 of the "independent" dataset
# The following models have been optimized (using tsai):
# InceptionTimePlus, LSTM, LSTM_FCN, MLSTM_FCN
# See "OnHW_DL_optimize.py" for more details
# Parameters below correspond to the ones optimized used in our paper
# Randomized nature of the optimization process may result in different parameters at different runs
DL_OPT_MODEL_LIST = [(InceptionTimePlus, {'lower': {'arch_params': {'nf': 64, 'fc_dropout': 0.3, 'depth': 6}, 'hyp_params': {'lr': 0.0004661310232858109, 'epoch': 25}},
                                        'upper': {'arch_params': {'nf': 56, 'fc_dropout': 0.5, 'depth': 6}, 'hyp_params': {'lr': 0.0034188037154318637, 'epoch': 25}},
                                        'both': {'arch_params': {'nf': 64, 'fc_dropout': 0, 'depth': 6}, 'hyp_params': {'lr': 0.0001365551896356981, 'epoch': 25}}}
                    ),
                    (LSTM, {'lower': {'arch_params': {'n_layers': 2, 'fc_dropout': 0.1, 'rnn_dropout': 0.6, 'bidirectional': False}, 'hyp_params': {'lr': 0.006592888138467693, 'epoch': 25}},
                            'upper': {'arch_params': {'n_layers': 2, 'fc_dropout': 0.0, 'rnn_dropout': 0.6, 'bidirectional': False}, 'hyp_params': {'lr': 0.004225695900534662, 'epoch': 25}},
                            'both': {'arch_params': {'n_layers': 3, 'fc_dropout': 0.1, 'rnn_dropout': 0.5, 'bidirectional': False}, 'hyp_params': {'lr': 0.004927556969650458, 'epoch': 25}}
                            }
                    ),
                    (LSTM_FCN, {'lower': {'arch_params': {'rnn_layers': 2, 'fc_dropout': 0.2, 'rnn_dropout': 0.6, 'bidirectional': False}, 'hyp_params': {'lr': 0.0016159798580482959, 'epoch': 25}},
                            'upper': {'arch_params': {'rnn_layers': 3, 'fc_dropout': 0.3, 'rnn_dropout': 0.7, 'bidirectional': False}, 'hyp_params': {'lr': 0.0035451611009339164, 'epoch': 25}},
                            'both': {'arch_params': {'rnn_layers': 2, 'fc_dropout': 0.4, 'rnn_dropout': 0.7, 'bidirectional': True}, 'hyp_params': {'lr': 0.0022930755787053435, 'epoch': 25}}
                            }
                    ),                    
                    (MLSTM_FCN, {'lower': {'arch_params': {'rnn_layers': 1, 'fc_dropout': 0.1, 'rnn_dropout': 0.3, 'bidirectional': True}, 'hyp_params': {'lr': 0.004448015287061512, 'epoch': 25}},
                            'upper': {'arch_params': {'rnn_layers': 2, 'fc_dropout': 0.4, 'rnn_dropout': 0.3, 'bidirectional': False}, 'hyp_params': {'lr': 0.0018710149343148215, 'epoch': 25}},
                            'both': {'arch_params': {'rnn_layers': 1, 'fc_dropout': 0.3, 'rnn_dropout': 0.4, 'bidirectional': True}, 'hyp_params': {'lr': 0.005565541649771458, 'epoch': 25}}
                            }
                    )
                    ]
                    


def create_DL_MODEL_LIST(case):
    temp = DL_MODEL_LIST+DL_OPT_MODEL_LIST
    return temp  
    
def OnHW_DL_get_prediction_space(case, dependency, k_fold_number, name):
    trainX, trainy, testX, testy = tsai_ready_data(case, dependency, k_fold_number)
    X, y, splits = combine_split_data([trainX, testX], [trainy, testy])

    test_actual = y[splits[1]]
    test_preds_by_model = {}
    prediction_rows = []
    first_name = ""
    x=True
    
    length = len(DL_MODEL_LIST)
    index = 0
    DL_MODEL_LIST2 = create_DL_MODEL_LIST(case)
    for i, (arch, k) in enumerate(DL_MODEL_LIST2):
        if index>=length:
            k_arch = k[case]['arch_params']
            k_hyp = k[case]['hyp_params']

            name = f"{arch.__name__}_{k_arch}_{k_hyp}_opt"
        else:
            name = f"{arch.__name__}_{k}"
        index = index+1
        learn = hf.OnHW_DL_load_learn_by_name(case, dependency, k_fold_number, name)
        test_probas, test_targets, test_preds = learn.get_X_preds(X[splits[1]], test_actual, with_decoded=True)
        test_preds_by_model[name] = test_preds
        if x:
            first_name = name
            x = False
    for test_num in range(len(test_preds_by_model[first_name])):
        pred_row = {"actual":test_actual[test_num]}
        
        length = len(DL_MODEL_LIST)
        index = 0
        for i, (arch, k) in enumerate(DL_MODEL_LIST2):
            if index>=length:
                k_arch = k[case]['arch_params']
                k_hyp = k[case]['hyp_params']

                name = f"{arch.__name__}_{k_arch}_{k_hyp}_opt"
            else:
                name = f"{arch.__name__}_{k}"
            index = index+1
            pred_row[name] = test_preds_by_model[name][test_num]
        prediction_rows.append(pred_row)

    return prediction_rows


def OnHW_DL_analyze_failure_space_all():
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                name = f"failure_space"
                print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{name}")
                conf_mat_csv_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"prediction_rows{case}_{dependency}_{k_fold_number}.csv"))
                try:
                    prediction_rows = pd.read_csv(conf_mat_csv_file)
                except:
                    OnHW_DL_get_prediction_space_all()
                    prediction_rows = pd.read_csv(conf_mat_csv_file)
                OnHW_DL_get_failure_space(prediction_rows,case,dependency,k_fold_number)

def OnHW_DL_get_failure_space(prediction_rows,case,dependency,k_fold_number):
    
    length = len(DL_MODEL_LIST)
    index = 0
    DL_MODEL_LIST2 = create_DL_MODEL_LIST(case)
    for i, (arch, k) in enumerate(DL_MODEL_LIST2):
        if index>=length:
            k_arch = k[case]['arch_params']
            k_hyp = k[case]['hyp_params']

            name = f"{arch.__name__}_{k_arch}_{k_hyp}_opt"
        else:
            name = f"{arch.__name__}_{k}"
        index = index+1
        failure_space = []
        predictions = prediction_rows.to_dict('records')
        for row in predictions:
            if row["actual"]!= row[name]:
                failure_space.append(row.copy())

        failure_tally_for_bar_right = []
        failure_tally_for_bar_wrong = []
        bar_labels = []
        x = 1
        
        
        length = len(DL_MODEL_LIST)
        index1 = 0
        for i, (arch2, k2) in enumerate(DL_MODEL_LIST2):
            if index1>=length:
                k_arch2 = k2[case]['arch_params']
                k_hyp2 = k2[case]['hyp_params']

                name_2 = f"{arch2.__name__}_{k_arch2}_{k_hyp2}_opt"
            else:
                name_2 = f"{arch2.__name__}_{k2}"
            index1 = index1+1
            right = 0
            wrong = 0
            if (name == name_2):
                pass
            else:
                for row in failure_space:
                    if row["actual"]==row[name_2]:
                        right+=1
                    else:
                        wrong +=1
                failure_tally_for_bar_right.append(right)
                failure_tally_for_bar_wrong.append(wrong)
                bar_labels.append(str(x))
                x+=1
        
        conf_mat_csv_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"{case}_{dependency}_{k_fold_number}_{name}.pdf"))

        plt.bar(bar_labels, failure_tally_for_bar_right, color='tab:green')
        plt.bar(bar_labels, failure_tally_for_bar_wrong, bottom=failure_tally_for_bar_right, color='tab:red')
        title_name = name.replace("_","").replace("{","").replace("}","")
        plt.suptitle(f'{title_name} Failure Space')
        plt.xlabel('Model number')
        try:
            plt.savefig(conf_mat_csv_file, format = 'pdf',bbox_inches = 'tight')
        except:
            plt.savefig(conf_mat_csv_file.replace(":","").replace("'","_"), format = 'pdf',bbox_inches = 'tight')
        plt.clf()



def OnHW_DL_get_failure_space_array(prediction_rows,case,dependency,k_fold_number,letter):
    
    length = len(DL_MODEL_LIST)
    index = 0
    DL_MODEL_LIST2 = create_DL_MODEL_LIST(case)

    rows = []
    predictions = prediction_rows.to_dict('records')
    for row in predictions:
        # print(row)
        if row["actual"]== letter:
            # print("in")
            r = []
            
            
    for i, (arch, k) in enumerate(DL_MODEL_LIST2):
        if index>=length:
            k_arch = k[case]['arch_params']
            k_hyp = k[case]['hyp_params']

            name = f"{arch.__name__}_{k_arch}_{k_hyp}_opt"
        else:
            name = f"{arch.__name__}_{k}"
        index = index+1

        if row[name] == row["actual"]:
            r.append(1)
        else:
            r.append(0)
            rows.append(r)
    df = pd.DataFrame(rows).transpose()
    plt.figure()
    plt.imshow(df, cmap='hot', interpolation='nearest')
    letter_case = "0"
    if letter in ascii_lowercase:
        letter_case = "1"
    conf_mat_csv_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"prediction_space_{case}_{dependency}_{k_fold_number}_{letter}_{letter_case}.pdf"))

    plt.suptitle(f'{letter} Prediction Space', fontsize=10)
    plt.ylabel('Model number')
    plt.xlabel('Test data point')
    plt.savefig(conf_mat_csv_file, format = 'pdf',bbox_inches = 'tight')
    plt.clf() 

def OnHW_DL_get_prediction_space_all():
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                name = f"failure_space"
                print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{name}")
                prediction_rows= OnHW_DL_get_prediction_space(case, dependency, k_fold_number, name)
                df_results = pd.DataFrame(prediction_rows)
                conf_mat_csv_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"prediction_rows{case}_{dependency}_{k_fold_number}.csv"))
                df_results.to_csv(conf_mat_csv_file, index=False)

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
    return pad_sequences(X, maxlen=maxlen, truncating='post')
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

import pathlib 
if __name__ == "__main__":
    plt = platform.system()
    if plt == 'Windows': 
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        temp = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PosixPath
    #
    '''
    Precursor to failure space analysis (only needs to be run once). Must be run for all DL_MODEL_LIST models before failurespace is run
    
    '''
    OnHW_DL_get_prediction_space_all()

    '''
    Runs failure space analysis for all models listed
    '''
    OnHW_DL_analyze_failure_space_all()


    pathlib.PosixPath = temp

