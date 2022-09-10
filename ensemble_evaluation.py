from pickle import TRUE
from string import ascii_letters, ascii_lowercase, ascii_uppercase
from utils import config
from utils import folders_and_files
from platform import system

import os
import pandas as pd
import numpy as np

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tsai.all import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

 
# BOTH_DL_MODEL_LIST = []

def create_DL_MODEL_LIST(case):
    temp = DL_MODEL_LIST+DL_OPT_MODEL_LIST
    return temp  

def OnHW_DL_evaluate_all_voted(voting_method = "weighted_sum"):
    path_to_results = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, config.DL_RESULTS_CSV))
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                name = f"best_voted"
                print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{name}")
                accuracy, report, conf_mat = OnHW_DL_evaluate_model_vote(case, dependency, k_fold_number, name,voting_method)
                L = [[case, dependency, k_fold_number, name, accuracy]]
                df_results = pd.DataFrame(L, columns=['case', 'dependency', 'fold', 'model', 'accuracy'])
                if os.path.exists(path_to_results):
                    df_results.to_csv(path_to_results, mode = 'a', index=False, header=False)
                else:
                    df_results.to_csv(path_to_results, index = False)

                classification_report_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"classification_report_{case}_{dependency}_{k_fold_number}_{name}.txt"))
                with open(classification_report_file, 'w') as f:
                    print(report, file=f)

                conf_mat_csv_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"conf_mat_{case}_{dependency}_{k_fold_number}_{name}.csv"))
                df_conf_mat = pd.DataFrame(conf_mat)
                df_conf_mat.to_csv(conf_mat_csv_file, index=False)

def OnHW_DL_evaluate_model_vote(case, dependency, k_fold_number, name,voting_method ="weighted_sum"):
    trainX, trainy, testX, testy = tsai_ready_data(case, dependency, k_fold_number)
    X, y, splits = combine_split_data([trainX, testX], [trainy, testy])

    test_actual = y[splits[1]]
    if case =='both':
        labels = ascii_uppercase+ascii_lowercase
    elif case == 'upper':
        labels = ascii_uppercase
    elif case=='lower':
        labels = ascii_lowercase
    test_preds_by_model = {}
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
        learn = hf.OnHW_DL_load_learn_by_name(case, dependency, k_fold_number, name)
        test_probas, test_targets, test_preds = learn.get_X_preds(X[splits[1]], test_actual, with_decoded=True)
        if "sum" in voting_method:
            test_preds_by_model[name] = test_probas
        else:
            test_preds_by_model[name] = test_preds
        index =index+1
    # test_preds = vote_for_class(weighted, probability, test_actual, labels, test_preds_by_model)
    test_preds = vote_for_class_v2(test_actual, labels, test_preds_by_model,case,voting_method)
    conf_mat = confusion_matrix(test_actual, test_preds)
    report = classification_report(test_actual, test_preds, zero_division=1, digits=4)
    accuracy = accuracy_score(test_actual, test_preds)

    return accuracy, report, conf_mat

def vote_for_class_v2(test_actual, labels, test_preds_by_model,case,voting_method = "weighted_sum"):
    test_preds = []
    
    for i in range(len(test_actual)):
        monomial = {}
        for letter in range(len(labels)):
            monomial[labels[letter]]={}

        for name in test_preds_by_model:
            guess = test_preds_by_model[name][i]
            for letter in range(len(labels)):
                monomial[labels[letter]][name] = guess[letter]
        test_preds.append(make_prediction_using_monomial(monomial,method,case=case))

    return test_preds

def make_prediction_using_monomial(monomial,ordering,case):
    if ordering =="lex":
        return ordering_lexographic(monomial,case)
    elif ordering == "gr-lex":
        return ordering_graded_lexographic(monomial,case)
    elif ordering == "weighted_sum":
        return ordering_weighted_sum(monomial)
    elif ordering == "weighted_vote":
        return ordering_weighted_vote(monomial)
    elif ordering == "majority":
        return ordering_majority_voting(monomial)
    elif ordering == "sum_proba":
        return ordering_sum_probability_voting(monomial)

def ordering_lexographic(monomial,case):
    #First we determine Inception_time list keep track of ties
    #Next we take all of the ties, and check them in xception_time and so on
    ties = []
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
        for letter in letters_to_try:
            proba = monomial[letter][name]
            if ties ==[]:
                ties = [(letter,proba)]
            else:
                if ties[0][1]<proba:
                    ties = [(letter,proba)]
                elif ties[0][1]==proba:
                    ties.append((letter,proba))
                else:
                    pass
        if len(ties)>1:
            letters_to_try = []
            for l in ties:
                letters_to_try.append(l[0])
        else:
            return ties[0][0]


def ordering_sum_probability_voting(monomial):
    probabilities_sum={}
    for letter in monomial:
        probabilities_sum[letter] = 0
        for model in monomial[letter]:
            probabilities_sum[letter]+= monomial[letter][model]
    return max(probabilities_sum, key=probabilities_sum.get)
    

def ordering_weighted_sum(monomial):
    probabilities_sum={}
    for letter in monomial:
        probabilities_sum[letter] = 0
        for model in monomial[letter]:
            weight = weight_model(model)
            probabilities_sum[letter]+= monomial[letter][model]*weight
    return max(probabilities_sum, key=probabilities_sum.get)

def ordering_graded_lexographic(monomial,case):
    probabilities_sum={}
    for letter in monomial:
        probabilities_sum[letter] = 0
        for model in monomial[letter]:
            weight = weight_model(model)
            probabilities_sum[letter]+= monomial[letter][model]*weight

    max_value = max(numbers.values())
    max_letters = [k for k,v in numbers.items() if v == max_value]
    if len(max_letters)==1:
        return max_letters[0]
    ties = []
    letters_to_try = max_letters
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
        for letter in letters_to_try:
            proba = monomial[letter][name]
            if ties ==[]:
                ties = [(letter,proba)]
            else:
                if ties[0][1]<proba:
                    ties = [(letter,proba)]
                elif ties[0][1]==proba:
                    ties.append((letter,proba))
                else:
                    pass
        if len(ties)>1:
            letters_to_try = []
            for l in ties:
                letters_to_try.append(l[0])
        else:
            return ties[0][0]

def weight_model(model):


    '''
    The following function for determining weights was also tested (Before including optimizations).
    The weights are a scaling applied to the WI-L accuracies of the models. 
    This was also tested with the other datasets, however, all with worse results than what was used.

    range = [1,100]
    accuracies = {"FCN_{}":0.737,
                "InceptionTime_{}":0.834,
                "LSTM_FCN_{'bidirectional': True}":0.737,
                "LSTM_FCN_{}":0.744,
                "LSTM_{'n_layers': 2, 'bidirectional': False}":0.783,
                "LSTM_{'n_layers': 2, 'bidirectional': True}":0.785,
                "MLSTM_FCN_{'bidirectional': True}":0.745,
                "MLSTM_FCN_{}":0.742,
                "ResCNN_{}":0.783,
                "ResNet_{}":0.816,
                "XCM_{}":0.724,
                "XceptionTime_{}":0.829}
    old_min = min(accuracies.values())
    old_max = max(accuracies.values())
    weight = (((accuracies[model] - old_min) * (range[1] - range[0])) / (old_max - old_min)) + range[0]
    '''
    ###MUST CHANGE THESE NAMES
    accuracies = {"FCN_{}":1,
                "InceptionTime_{}":45,
                "InceptionTime_{'nf': 64, 'depth': 12}":45,
                "InceptionTimePlus_{'nf': 56, 'fc_dropout': 0.5, 'depth': 6}_{'lr': 0.0034188037154318637, 'epoch': 25}_opt":45,
                "InceptionTimePlus_{'nf': 64, 'fc_dropout': 0.3, 'depth': 6}_{'lr': 0.0004661310232858109, 'epoch': 25}_opt":45,
                "InceptionTimePlus_{'nf': 64, 'fc_dropout': 0, 'depth': 6}_{'lr': 0.0001365551896356981, 'epoch': 25}_opt":45,

                "LSTM_{'n_layers': 2, 'fc_dropout': 0.0, 'rnn_dropout': 0.6, 'bidirectional': False}_{'lr': 0.004225695900534662, 'epoch': 25}_opt":9,
                "LSTM_{'n_layers': 2, 'fc_dropout': 0.1, 'rnn_dropout': 0.6, 'bidirectional': False}_{'lr': 0.006592888138467693, 'epoch': 25}_opt":9,
                "LSTM_{'n_layers': 3, 'fc_dropout': 0.1, 'rnn_dropout': 0.5, 'bidirectional': False}_{'lr': 0.004927556969650458, 'epoch': 25}_opt":9,

                "LSTM_FCN_{'rnn_layers': 3, 'fc_dropout': 0.3, 'rnn_dropout': 0.7, 'bidirectional': False}_{'lr': 0.0035451611009339164, 'epoch': 25}_opt":9,
                "LSTM_FCN_{'rnn_layers': 2, 'fc_dropout': 0.2, 'rnn_dropout': 0.6, 'bidirectional': False}_{'lr': 0.0016159798580482959, 'epoch': 25}_opt":9,
                "LSTM_FCN_{'rnn_layers': 2, 'fc_dropout': 0.4, 'rnn_dropout': 0.7, 'bidirectional': True}_{'lr': 0.0022930755787053435, 'epoch': 25}_opt":9,

                "MLSTM_FCN_{'rnn_layers': 2, 'fc_dropout': 0.4, 'rnn_dropout': 0.3, 'bidirectional': False}_{'lr': 0.0018710149343148215, 'epoch': 25}_opt":9,
                "MLSTM_FCN_{'rnn_layers': 1, 'fc_dropout': 0.1, 'rnn_dropout': 0.3, 'bidirectional': True}_{'lr': 0.004448015287061512, 'epoch': 25}_opt":9,
                "MLSTM_FCN_{'rnn_layers': 1, 'fc_dropout': 0.3, 'rnn_dropout': 0.4, 'bidirectional': True}_{'lr': 0.005565541649771458, 'epoch': 25}_opt":9,

                "LSTM_FCN_{'bidirectional': True}":1,
                "LSTM_FCN_{}":1,
                "LSTM_{'n_layers': 2, 'bidirectional': False}":1,
                "LSTM_{'n_layers': 2, 'bidirectional': True}":1,
                "MLSTM_FCN_{'bidirectional': True}":1,
                "MLSTM_FCN_{}":1,
                "ResCNN_{}":9,
                "ResNet_{}":45,
                "XCM_{}":1,
                "XceptionTime_{}":45} 
    return accuracies[model]


def ordering_majority_voting(monomial):
    highest_prediction = {}
    key = list(monomial.keys())[0]
    for model in monomial[key]:
        highest_prediction[model] = (key,monomial[key][model])
    for letter in monomial:
        for model in monomial[letter]:
            if monomial[letter][model]>highest_prediction[model][1]:
                highest_prediction[model] = (letter,monomial[letter][model])

    track = {}
    for key,value in highest_prediction.items():
        if value[0] not in track:
            track[value[0]]=0
        else:
            track[value[0]]+=1

    return max(track, key=track.get)
    
def ordering_weighted_vote(monomial):
    highest_prediction = {}
    key = list(monomial.keys())[0]
    for model in monomial[key]:
        highest_prediction[model] = (key,monomial[key][model])
    for letter in monomial:
        for model in monomial[letter]:
            if monomial[letter][model]>highest_prediction[model][1]:
                highest_prediction[model] = (letter,monomial[letter][model])

    track = {}
    for model in highest_prediction:
        value = highest_prediction[model]
        if value[0] not in track:
            track[value[0]]=weight_model(model)
        else:
            track[value[0]]+=weight_model(model)

    return max(track, key=track.get)


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


import pathlib 
if __name__ == "__main__":
    plt = platform.system()
    if plt == 'Windows': 
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        temp = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PosixPath
    '''
        The following function runs the ensemble method as specified in the parameter. All models included in the ensemble should be trained beforehand

        The following can be taken as parameter:
        "weighted_sum": for the weighted sum approach as specified in the paper. Weightings are pre selected in the weight_model function
        "weighted_vote": for the weighted voting approach as specified in the paper. Weightings are pre selected in the weight_model function
        "majority": for an unweighted voting method
        "sum_proba": for a soft voting method.
        "lex" for lexographic ordering. The models are ordered by their position in DL_MODEL_LIST
       '''
    #Evaluate ensemble models (only run after models have been trained)
    OnHW_DL_evaluate_all_voted("weighted_sum")

    if plt == 'Windows': 
        pathlib.PosixPath = temp
    else:
        pathlib.WindowsPath = temp