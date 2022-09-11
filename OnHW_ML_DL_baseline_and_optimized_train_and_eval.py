from utils import config
from utils import folders_and_files

import os
import pandas as pd
import numpy as np

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from tsai.all import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

# A list of ML baseline models
ML_MODEL_AND_SCALER_DICT = {'SVM_rbf': [SVC(kernel='rbf', decision_function_shape='ovo', random_state=config.RANDOM_STATE), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'SVM_linear': [SVC(kernel='linear', decision_function_shape='ovo', random_state=config.RANDOM_STATE), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'RFC': [RandomForestClassifier(criterion='gini', n_estimators=100, random_state=config.RANDOM_STATE), None],
                         'DT': [DecisionTreeClassifier(criterion='gini', random_state=config.RANDOM_STATE), None],
                         'ET': [ExtraTreesClassifier(criterion='gini', n_estimators=100, random_state=config.RANDOM_STATE), None],
                         'kNN': [KNeighborsClassifier(), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'LogReg': [LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         }

# A list of ML models for metric learning
METRICLEARNING_MODEL_AND_SCALER_DICT = {
                         'metriclearn_SVM_rbf': [SVC(kernel='rbf', decision_function_shape='ovo', random_state=config.RANDOM_STATE), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'metriclearn_SVM_linear': [SVC(kernel='linear', decision_function_shape='ovo', random_state=config.RANDOM_STATE), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'metriclearn_RFC': [RandomForestClassifier(criterion='gini', n_estimators=100, random_state=config.RANDOM_STATE), None],
                         'metriclearn_DT': [DecisionTreeClassifier(criterion='gini', random_state=config.RANDOM_STATE), None],
                         'metriclearn_ET': [ExtraTreesClassifier(criterion='gini', n_estimators=100, random_state=config.RANDOM_STATE), None],
                         'metriclearn_kNN': [KNeighborsClassifier(), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
                         'metriclearn_LogReg': [LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)],
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

# {'case': [lower, upper, both], 'dependency': [dep, indep], 'k_fold_number': [0,1,2,3,4]}
def load_OnHW_data(case, dependency, k_fold_number):
    path_to_folder = f"onhw2_{case}_{dependency}_{k_fold_number}"
    train_X = np.load(os.path.join(config.PATH_TO_PREPEND, path_to_folder, "X_train.npy"), allow_pickle=True)
    test_X = np.load(os.path.join(config.PATH_TO_PREPEND, path_to_folder, "X_test.npy"), allow_pickle=True)
    train_y = np.load(os.path.join(config.PATH_TO_PREPEND, path_to_folder, "y_train.npy"), allow_pickle=True)
    test_y = np.load(os.path.join(config.PATH_TO_PREPEND, path_to_folder, "y_test.npy"), allow_pickle=True)
    return train_X, train_y, test_X, test_y

def filter_train_test(X, y, case):
    lower_bound, upper_bound = max(config.HARD_LOWER_BOUND, config.MEAN[case] - config.CUT_OFF_COEFF*config.STD[case]), min(config.HARD_UPPER_BOUND, config.MEAN[case] + config.CUT_OFF_COEFF*config.STD[case])
    mask = [(len(data) >= lower_bound) & (len(data) <= upper_bound) for data in X]
    return X[mask], y[mask]

def X_npy_to_df(X):
    df = pd.DataFrame(columns = ['id', 'time', 'f_0', 'f_1', 'f_2','f_3','f_4','f_5','f_6','f_7','f_8','f_9','f_10','f_11','f_12'])

    id = 0
    for data in X:
        little_df = pd.DataFrame(columns = ['id', 'time', 'f_0', 'f_1', 'f_2','f_3','f_4','f_5','f_6','f_7','f_8','f_9','f_10','f_11','f_12'])

        time_steps = len(data)
        little_df['time'] = range(time_steps)

        little_df.iloc[:, -13:] = pd.DataFrame(data)
        little_df['id'] = id
        id+=1

        df = pd.concat([df, little_df])

    return df.reset_index(drop=True)

def ts_extract_feautures(train_X, train_y, test_X):
    df_train_X = X_npy_to_df(train_X)
    df_test_X = X_npy_to_df(test_X)

    extracted_features = extract_features(df_train_X, column_id='id', column_sort="time")
    # remove all NaN values
    impute(extracted_features)
    features_filtered_train = select_features(extracted_features, train_y, multiclass=True, n_significant=config.NUM_SIG)

    features_filtered_test = extract_features(df_test_X, column_id='id', column_sort="time",
                                               kind_to_fc_parameters=settings.from_columns(features_filtered_train.columns))

    features_filtered_test = features_filtered_test[features_filtered_train.columns]
    # remove all NaN values
    impute(features_filtered_test)

    return extracted_features, features_filtered_train, features_filtered_test

'''For each train and test OnHW data, the data is filtered and
statistical features are extracted using tsfresh. Corresponding outputs are stored under the 'output' folder.
The code assumes that the 'ouput' folder already exists.
The OnHW dataset (https://stabilodigital.com/onhw-dataset/) should be unzipped and placed under the 'dataset' folder so it looks like
dataset/
└── IMWUT_OnHW-chars_dataset_2021-06-30
    └── OnHW-chars_2021-06-30
        ├── onhw2_both_dep_0
        │   ├── X_test.npy
        │   ├── X_train.npy
        │   ├── y_test.npy
        │   └── y_train.npy
        ├── onhw2_both_dep_1
        ├── ...
        ├── ...        
        ├── onhw2_upper_indep_4
        │   ├── X_test.npy
        │   ├── X_train.npy
        │   ├── y_test.npy
        │   └── y_train.npy
        └── readme.txt
'''
def OnHW_ML_filter_and_extract():
    ## folders_and_files.make_folder_at(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
    path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                print(f"Processing dataset:{case}_{dependency}_{k_fold_number}")
                folder_name = f"{case}_{dependency}_{k_fold_number}"
                ## folders_and_files.make_folder_at(path_to_models_and_data, folder_name)

                train_X, train_y, test_X, test_y = load_OnHW_data(case, dependency, k_fold_number)
                train_X_filtered, train_y_filtered = filter_train_test(train_X, train_y, case)
                test_X_filtered, test_y_filtered = filter_train_test(test_X, test_y, case)

                extracted_features, features_filtered_train, features_filtered_test = ts_extract_feautures(
                    train_X_filtered, train_y_filtered, test_X_filtered)

                path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
                folder_name = f"{case}_{dependency}_{k_fold_number}"
                ## folders_and_files.make_folder_at(path_to_models_and_data, folder_name)

                np.save(os.path.join(path_to_models_and_data, folder_name, 'train_X_filtered.npy'), train_X_filtered)
                np.save(os.path.join(path_to_models_and_data, folder_name, 'train_y_filtered.npy'), train_y_filtered)
                np.save(os.path.join(path_to_models_and_data, folder_name, 'test_X_filtered.npy'), test_X_filtered)
                np.save(os.path.join(path_to_models_and_data, folder_name, 'test_y_filtered.npy'), test_y_filtered)
                features_filtered_train.to_csv(
                    os.path.join(path_to_models_and_data, folder_name, 'features_filtered_train.csv'), index=False)
                features_filtered_test.to_csv(
                    os.path.join(path_to_models_and_data, folder_name, 'features_filtered_test.csv'), index=False)


def OnHW_ML_read_filtered_data_and_extracted_features(case, dependency, k_fold_number):
    path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
    folder_name = f"{case}_{dependency}_{k_fold_number}"

    train_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_X_filtered.npy"), allow_pickle=True)
    train_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_y_filtered.npy"), allow_pickle=True)
    test_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_X_filtered.npy"), allow_pickle=True)
    test_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_y_filtered.npy"), allow_pickle=True)
    features_filtered_train = pd.read_csv(os.path.join(path_to_models_and_data, folder_name, "features_filtered_train.csv"))
    features_filtered_test = pd.read_csv(os.path.join(path_to_models_and_data, folder_name, "features_filtered_test.csv"))

    return train_X_filtered, train_y_filtered, test_X_filtered, test_y_filtered, features_filtered_train, features_filtered_test

def OnHW_ML_train_and_save(case, dependency, k_fold_number, model_and_scaler_dict):
    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(
        case, dependency, k_fold_number)

    for name, model_and_scaler in model_and_scaler_dict.items():
        print(f"[INFO] Training: {case}_{dependency}_{k_fold_number}_{name}")
        folder_name = f"{case}_{dependency}_{k_fold_number}"
        path_to_model = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA, folder_name)
        model_name, scaler_name = f"model_{name}", f"scaler_{name}"
        model, scaler = model_and_scaler[0], model_and_scaler[1]

        if scaler==None:
            train_X_features_transformed = train_X_features.copy()
        else:
            train_X_features_transformed = scaler.fit_transform(train_X_features)
            folders_and_files.save_model(path_to_model, scaler_name, scaler)

        model.fit(train_X_features_transformed, train_y)
        folders_and_files.save_model(path_to_model, model_name, model)

def OnHW_MetricLearn_train_and_save(case, dependency, k_fold_number, model_and_scaler_dict):
    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(
        case, dependency, k_fold_number)

    train_X_features.sort_index(axis=1, inplace=True)

    for name, model_and_scaler in model_and_scaler_dict.items():
        print(f"[INFO] Training: {case}_{dependency}_{k_fold_number}_{name}")
        folder_name = f"{case}_{dependency}_{k_fold_number}"
        path_to_model = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA, folder_name)
        model_name, scaler_name, nca_name = f"nca_model_{name}", f"nca_scaler_{name}", f"nca_{name}"
        model = model_and_scaler[0]

        scaler = StandardScaler() # NCA requires scaled data
        nca = NeighborhoodComponentsAnalysis(n_components=21, random_state=config.RANDOM_STATE)

        train_X_features_transformed = scaler.fit_transform(train_X_features)
        train_X_features_transformed = pd.DataFrame(train_X_features_transformed)
        nca.fit(train_X_features_transformed, train_y)
        train_X_features_transformed = nca.transform(train_X_features_transformed)
        
        folders_and_files.save_model(path_to_model, scaler_name, scaler)
        folders_and_files.save_model(path_to_model, nca_name, nca)

        model.fit(train_X_features_transformed, train_y)
        folders_and_files.save_model(path_to_model, model_name, model)

def OnHW_ML_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, name):
    folder_name = f"{case}_{dependency}_{k_fold_number}"
    path_to_model = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA, folder_name)
    model_name, scaler_name = f"model_{name}", f"scaler_{name}"
    model = folders_and_files.load_model(path_to_model, model_name)
    scaler = folders_and_files.load_model(path_to_model, scaler_name)

    return model, scaler

def OnHW_MetricLearn_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, name):
    folder_name = f"{case}_{dependency}_{k_fold_number}"
    path_to_model = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA, folder_name)
    model_name, scaler_name, nca_name = f"nca_model_{name}", f"nca_scaler_{name}", f"nca_{name}"
    model = folders_and_files.load_model(path_to_model, model_name)
    scaler = folders_and_files.load_model(path_to_model, scaler_name)
    nca = folders_and_files.load_model(path_to_model, nca_name)

    return model, scaler, nca

def OnHW_ML_evaluate_model(case, dependency, k_fold_number, name):
    model, scaler = OnHW_ML_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, name)

    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(
        case, dependency, k_fold_number)

    if scaler==None:
        test_X_features_transformed = test_X_features.copy()
    else:
        test_X_features_transformed = scaler.transform(test_X_features)

    preds = model.predict(test_X_features_transformed)
    report = classification_report(test_y, preds, zero_division=1, digits=4)
    conf_mat = confusion_matrix(test_y, preds)
    accuracy = accuracy_score(test_y, preds)

    return accuracy, report, conf_mat

def OnHW_MetricLearn_evaluate_model(case, dependency, k_fold_number, name):
    model, scaler, nca = OnHW_MetricLearn_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, name)

    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(
        case, dependency, k_fold_number)

    test_X_features.sort_index(axis=1, inplace=True)

    test_X_features_transformed = scaler.transform(test_X_features)
    test_X_features_transformed = pd.DataFrame(test_X_features_transformed)
    test_X_features_transformed = nca.transform(test_X_features_transformed)

    preds = model.predict(test_X_features_transformed)
    report = classification_report(test_y, preds, zero_division=1, digits=4)
    conf_mat = confusion_matrix(test_y, preds)
    accuracy = accuracy_score(test_y, preds)

    return accuracy, report, conf_mat

def OnHW_ML_train_all():
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                OnHW_ML_train_and_save(case, dependency, k_fold_number, ML_MODEL_AND_SCALER_DICT)

def OnHW_MetricLearn_train_all():
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                OnHW_MetricLearn_train_and_save(case, dependency, k_fold_number, METRICLEARNING_MODEL_AND_SCALER_DICT)

def OnHW_ML_evaluate_all():
    path_to_results = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, config.ML_RESULTS_CSV)
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                for name in ML_MODEL_AND_SCALER_DICT.keys():
                    print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{name}")
                    accuracy, report, conf_mat = OnHW_ML_evaluate_model(case, dependency, k_fold_number, name)
                    L = [[case, dependency, k_fold_number, name, accuracy]]
                    df_results = pd.DataFrame(L, columns=['case', 'dependency', 'fold', 'model', 'accuracy'])
                    if os.path.exists(path_to_results):
                        df_results.to_csv(path_to_results, mode = 'a', index=False, header=False)
                    else:
                        df_results.to_csv(path_to_results, index = False)

                    classification_report_file = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"classification_report_{case}_{dependency}_{k_fold_number}_{name}.txt")
                    with open(classification_report_file, 'w') as f:
                        print(report, file=f)

                    conf_mat_csv_file = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"conf_mat_{case}_{dependency}_{k_fold_number}_{name}.csv")
                    df_conf_mat = pd.DataFrame(conf_mat)
                    df_conf_mat.to_csv(conf_mat_csv_file, index=False)

def OnHW_MetricLearn_evaluate_all():
    path_to_results = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, config.ML_RESULTS_CSV)
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                for name in METRICLEARNING_MODEL_AND_SCALER_DICT.keys():
                    print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{name}")
                    accuracy, report, conf_mat = OnHW_MetricLearn_evaluate_model(case, dependency, k_fold_number, name)
                    L = [[case, dependency, k_fold_number, name, accuracy]]
                    df_results = pd.DataFrame(L, columns=['case', 'dependency', 'fold', 'model', 'accuracy'])
                    if os.path.exists(path_to_results):
                        df_results.to_csv(path_to_results, mode = 'a', index=False, header=False)
                    else:
                        df_results.to_csv(path_to_results, index = False)

                    classification_report_file = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"classification_report_{case}_{dependency}_{k_fold_number}_{name}.txt")
                    with open(classification_report_file, 'w') as f:
                        print(report, file=f)

                    conf_mat_csv_file = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"conf_mat_{case}_{dependency}_{k_fold_number}_{name}.csv")
                    df_conf_mat = pd.DataFrame(conf_mat)
                    df_conf_mat.to_csv(conf_mat_csv_file, index=False)

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


def OnHW_DL_optimized_train_and_save(case, dependency, k_fold_number, model_list):
    trainX, trainy, testX, testy = tsai_ready_data(case, dependency, k_fold_number)
    X, y, splits = combine_split_data([trainX, testX], [trainy, testy])

    batch_tfms = TSStandardize(by_sample=True)
    for i, (arch, K) in enumerate(model_list):

        k_arch = K[case]['arch_params']
        k_hyp = K[case]['hyp_params']
        epoch_num = k_hyp['epoch']
        lr = k_hyp['lr']

        name = f"{arch.__name__}_{k_arch}_{k_hyp}"
        print(f"[INFO] Training: {case}_{dependency}_{k_fold_number}_{name}")
        learn = TSClassifier(
        X, y, splits=splits, bs=[64, 128], batch_tfms=batch_tfms,
        arch=arch, arch_config=k_arch,
        metrics=accuracy)
        learn.fit_one_cycle(epoch_num, lr_max=lr)


        model_name = f"model_{name}_opt"
        path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.DL_MODELS_AND_DATA)
        folder_name = f"{case}_{dependency}_{k_fold_number}"
        ## folders_and_files.make_folder_at(path_to_models_and_data, folder_name)
        path_to_model = os.path.join(path_to_models_and_data, folder_name, f"{model_name}.pkl")
        learn.export(path_to_model)



def OnHW_DL_train_and_save(case, dependency, k_fold_number, model_list):
    trainX, trainy, testX, testy = tsai_ready_data(case, dependency, k_fold_number)
    X, y, splits = combine_split_data([trainX, testX], [trainy, testy])
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])

    for i, (arch, k) in enumerate(model_list):
        model = create_model(arch, dls=dls, **k)
        name = f"{arch.__name__}_{k}"
        print(f"[INFO] Training: {case}_{dependency}_{k_fold_number}_{name}")
        learn = Learner(dls, model,  metrics=accuracy)
        learn.fit_one_cycle(config.TSAI_EPOCH, 1e-3)

        model_name = f"model_{name}"
        path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.DL_MODELS_AND_DATA)
        folder_name = f"{case}_{dependency}_{k_fold_number}"
        ## folders_and_files.make_folder_at(path_to_models_and_data, folder_name)
        path_to_model = os.path.join(path_to_models_and_data, folder_name, f"{model_name}.pkl")
        learn.export(path_to_model)

def OnHW_DL_train_all():
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                OnHW_DL_train_and_save(case, dependency, k_fold_number, DL_MODEL_LIST)

def OnHW_DL_optimized_train_all():
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                OnHW_DL_optimized_train_and_save(case, dependency, k_fold_number, DL_OPT_MODEL_LIST)

def OnHW_DL_load_learn_by_name(case, dependency, k_fold_number, name):
    folder_name = f"{case}_{dependency}_{k_fold_number}"
    model_name = f"model_{name}.pkl"
    path_to_model = os.path.join(config.BASE_OUTPUT, config.DL_MODELS_AND_DATA, folder_name, model_name)

    learn = load_learner(path_to_model)

    return learn

def OnHW_DL_evaluate_model(case, dependency, k_fold_number, name):
    trainX, trainy, testX, testy = tsai_ready_data(case, dependency, k_fold_number)
    X, y, splits = combine_split_data([trainX, testX], [trainy, testy])

    learn = OnHW_DL_load_learn_by_name(case, dependency, k_fold_number, name)

    test_actual = y[splits[1]]
    test_probas, test_targets, test_preds = learn.get_X_preds(X[splits[1]], test_actual, with_decoded=True)

    conf_mat = confusion_matrix(test_actual, test_preds)
    report = classification_report(test_actual, test_preds, zero_division=1, digits=4)
    accuracy = accuracy_score(test_actual, test_preds)

    return accuracy, report, conf_mat



def OnHW_DL_evaluate_all():
    path_to_results = os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, config.DL_RESULTS_CSV)
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                for i, (arch, k) in enumerate(DL_MODEL_LIST):
                    name = f"{arch.__name__}_{k}"
                    print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{name}")
                    accuracy, report, conf_mat = OnHW_DL_evaluate_model(case, dependency, k_fold_number, name)
                    L = [[case, dependency, k_fold_number, name, accuracy]]
                    df_results = pd.DataFrame(L, columns=['case', 'dependency', 'fold', 'model', 'accuracy'])
                    if os.path.exists(path_to_results):
                        df_results.to_csv(path_to_results, mode = 'a', index=False, header=False)
                    else:
                        df_results.to_csv(path_to_results, index = False)

                    classification_report_file = os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"classification_report_{case}_{dependency}_{k_fold_number}_{name}.txt")
                    with open(classification_report_file, 'w') as f:
                        print(report, file=f)

                    conf_mat_csv_file = os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, f"conf_mat_{case}_{dependency}_{k_fold_number}_{name}.csv")
                    df_conf_mat = pd.DataFrame(conf_mat)
                    df_conf_mat.to_csv(conf_mat_csv_file, index=False)


def OnHW_DL_optimized_evaluate_all():
    path_to_results = os.path.join(config.BASE_OUTPUT, config.DL_RESULTS_OPT, config.DL_RESULTS_OPT_CSV)
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                for i, (arch, K) in enumerate(DL_OPT_MODEL_LIST):

                    k_arch = K[case]['arch_params']
                    k_hyp = K[case]['hyp_params']

                    name = f"{arch.__name__}_{k_arch}_{k_hyp}_opt"

                    print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{name}")
                    accuracy, report, conf_mat = OnHW_DL_evaluate_model(case, dependency, k_fold_number, name)
                    L = [[case, dependency, k_fold_number, name, accuracy]]
                    df_results = pd.DataFrame(L, columns=['case', 'dependency', 'fold', 'model', 'accuracy'])
                    if os.path.exists(path_to_results):
                        df_results.to_csv(path_to_results, mode = 'a', index=False, header=False)
                    else:
                        df_results.to_csv(path_to_results, index = False)

                    classification_report_file = os.path.join(config.BASE_OUTPUT, config.DL_RESULTS_OPT, f"classification_report_{case}_{dependency}_{k_fold_number}_{name}.txt")
                    with open(classification_report_file, 'w') as f:
                        print(report, file=f)

                    conf_mat_csv_file = os.path.join(config.BASE_OUTPUT, config.DL_RESULTS_OPT, f"conf_mat_{case}_{dependency}_{k_fold_number}_{name}.csv")
                    df_conf_mat = pd.DataFrame(conf_mat)
                    df_conf_mat.to_csv(conf_mat_csv_file, index=False)


def make_some_folders():
    folders_and_files.make_folder_at('.', config.BASE_OUTPUT)

    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.ML_RESULTS)
    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.DL_RESULTS)
    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.DL_RESULTS_OPT)
    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.OPTUNA)

    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.DL_MODELS_AND_DATA)

    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                folder_name = f"{case}_{dependency}_{k_fold_number}"
                path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
                folders_and_files.make_folder_at(path_to_models_and_data, folder_name)
                
                path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.DL_MODELS_AND_DATA)                
                folders_and_files.make_folder_at(path_to_models_and_data, folder_name)


if __name__ == "__main__":
    
    # Creating the "output" directory and several subdirectories for saving results and models
    print("[INFO] Creating the 'output' directory and several subdirectories for saving results and models")
    make_some_folders()

    '''Run OnHW_ML_filter_and_extract() only once. It takes long time to run through 3*2*5=30 folders
    Each folder is done in about 15 min for lower and upper, 45 min for combined/both ==> 14 hours, in total
    '''
    OnHW_ML_filter_and_extract()
    OnHW_ML_train_all()
    OnHW_ML_evaluate_all()

    '''Run OnHW_MetricLearn_train_all() and OnHW_MetricLearn_evaluate_all() to train and evaluate metric learning models
    Run OnHW_ML_filter_and_extract() prior to running Metric Learning algorithms to extract features
    '''
    OnHW_MetricLearn_train_all()
    OnHW_MetricLearn_evaluate_all()


    '''No need to run 'filtering functions' as filtered data has already been generated during OnHW_ML_filter_and_extract() above
    Also assumes the 'output' folder already exists (created in the beginning of the ML train/evaluate process above). 
    If not, either manually create it, or run folders_and_files.make_folder_at('.', config.BASE_OUTPUT)
    '''
    OnHW_DL_train_all()
    OnHW_DL_evaluate_all()

    '''The following trainings are based on models whose parameters have been optimized using tsai
    The following models are considered:
    InceptionTimePlus, LSTM, LSTM_FCN, MLSTM_FCN
    See the list "DL_OPT_MODEL_LIST" above and "OnHW_DL_optimize.py" for more details
    '''
    OnHW_DL_optimized_train_all() 
    OnHW_DL_optimized_evaluate_all()