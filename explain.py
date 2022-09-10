from pickle import TRUE
from string import ascii_letters, ascii_lowercase, ascii_uppercase
from utils import config
from utils import folders_and_files
from platform import system


import os
import numpy as np

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils.lime_timeseries import LimeTimeSeriesExplainer
from tsai.all import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import utils.helper_files as hf
from operator import itemgetter

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

def OnHW_explain_all(data_row = 0):
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                
                length = len(DL_MODEL_LIST)
                index = 0
                DL_MODEL_LIST2 = create_DL_MODEL_LIST(case)
                for i, (arch, k) in enumerate(DL_MODEL_LIST2):
                    print(length)
                    if index>=length:
                        k_arch = k[case]['arch_params']
                        k_hyp = k[case]['hyp_params']

                        name = f"{arch.__name__}_{k_arch}_{k_hyp}_opt"
                    else:
                        name = f"{arch.__name__}_{k}"
                    index = index+1
                    print(f"[INFO] Evaluating LIME: {case}_{dependency}_{k_fold_number}_{name}")
                    exp = OnHW_explain_Lime(case, dependency, k_fold_number, name,data_row=data_row)
                    lime_html_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"lime_{case}_{dependency}_{k_fold_number}_{name}.html"))
                    if (system() == 'Windows'):
                            lime_html_file = lime_html_file.replace("{","_").replace("}","_").replace(":","_").replace("'","")
                    exp.save_to_file(lime_html_file)

def learn_for_lime(self,x): # Have to add self since this will become a method
    y= np.transpose(x, (0, 1, 2))
    temp = self.get_X_preds(y)[0]
    temp2 = temp.numpy()
    return temp2
fastai.learner.Learner.learn_lime = learn_for_lime

def OnHW_explain_Lime(case, dependency, k_fold_number, name,scaled = True,data_row = 1):
    trainX, trainy, testX, testy = tsai_ready_data(case, dependency, k_fold_number)
    X, y, splits = combine_split_data([trainX, testX], [trainy, testy])

    learn = hf.OnHW_DL_load_learn_by_name(case, dependency, k_fold_number, name)

    test_actual = y[splits[1]]
    channel_names = ["%d" % x for x in range(13)]
    explainer = LimeTimeSeriesExplainer(class_names=list(ascii_lowercase),signal_names= channel_names)
    data_row = data_row
    x= np.transpose(X[splits[1]][data_row], (0, 1))
    # test_probas, test_targets, test_preds = learn.get_X_preds(X[splits[1]], test_actual, with_decoded=True)
    prediction = learn.get_X_preds(X[splits[1]][data_row].reshape(1,13,104),test_actual[data_row], with_decoded=True)[2][0]

    num_slices = 20
    num_features = 30
    exp = explainer.explain_instance(x, learn.learn_lime, num_slices = num_slices, num_features=num_features,
                                 replacement_method='total_mean')
    #Plot bar
    exp.as_pyplot_figure()
    lime_png_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"lime_bar_{case}_{dependency}_{k_fold_number}_{name}_{data_row}_{test_actual[data_row]}_{prediction}.pdf"))
    if (system() == 'Windows'): lime_png_file = lime_png_file.replace(",","_").replace(":","")
    plt.savefig(lime_png_file, format = 'pdf',bbox_inches = 'tight')
    # plt.show()
    plt.clf()

    #Plot line with bars overlay
    values_per_slice = math.ceil(x.shape[1]/num_slices)

    plt.clf()
    fig, axes = plt.subplots(13)
    title_name = name.replace("_","").replace("{","").replace("}","")
    # title_name = "LSTM Optimized"
    plt.suptitle(f'{title_name} Explanation')
    # fig.suptitle(f'{case}_{dependency}_{k_fold_number}_{name}')
    ax =0
    if scaled:
        maximum = max(exp.as_list(),key=itemgetter(1))[1]
        scaled = 1 / maximum
    else:
        scaled = 2
    for chan in channel_names:
        axes[ax].plot(x[ax,:], color='b', label='Explained instance')
        # axes[ax].yaxis.set_visible(False)
        axes[ax].set_yticklabels([])
        axes[ax].set_yticks([])
        axes[ax].set_ylabel(f"Channel {chan}",rotation=0, labelpad=45)
        for i in range(num_features):
            feature_and_channel, weight = exp.as_list()[i]
            feature_str, channel = feature_and_channel.split(" - ")
            feature = int(feature_str)
            start = feature * values_per_slice
            end = start + values_per_slice
            color = 'tab:red' if weight < 0 else 'tab:green' 
            
            if chan==channel:
                axes[ax].axvspan(start, end, color=color, alpha=min(abs(scaled*weight),1))
        ax+=1
    lime_ts_path_file = hf.path_to_windows(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"lime_ts_stacked_{case}_{dependency}_{k_fold_number}_{name}_{data_row}_{test_actual[data_row]}_{prediction}.pdf"))
    if (system() == 'Windows'): lime_ts_path_file = lime_ts_path_file.replace(",","_").replace(":","")
    plt.savefig(lime_ts_path_file, format = 'pdf',bbox_inches = 'tight')
    plt.clf()
    exp.as_html()
    return exp


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
def OnHW_ML_read_filtered_data(case, dependency, k_fold_number):
    path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
    folder_name = f"{case}_{dependency}_{k_fold_number}"

    train_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_X_filtered.npy"), allow_pickle=True)
    train_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_y_filtered.npy"), allow_pickle=True)
    test_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_X_filtered.npy"), allow_pickle=True)
    test_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_y_filtered.npy"), allow_pickle=True)

    return train_X_filtered, train_y_filtered, test_X_filtered, test_y_filtered

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
    Will create lime based explanation images for all models specified in DL_MODEL_LIST
    The data_row specified as a parameter corresponds to the element of the testing data to be explained
    '''
    #Lime explanation analysis
    OnHW_explain_all()

    if plt == 'Windows': 
        pathlib.PosixPath = temp
    else:
        pathlib.WindowsPath = temp