from importlib.resources import path

from utils import config
from utils import folders_and_files

import os
import pandas as pd
import numpy as np
import pickle

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Produces Figure 4, comparing the performance of KNN over various levels of n_components


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
    df = pd.DataFrame(columns = ['id',  'time', 'f_0', 'f_1', 'f_2','f_3','f_4','f_5','f_6','f_7','f_8','f_9','f_10','f_11','f_12'])

    id = 0
    for data in X:
        little_df = pd.DataFrame(columns = ['id',  'time', 'f_0', 'f_1', 'f_2','f_3','f_4','f_5','f_6','f_7','f_8','f_9','f_10','f_11','f_12'])

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
    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
    path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                print(f"Processing dataset:{case}_{dependency}_{k_fold_number}")
                folder_name = f"{case}_{dependency}_{k_fold_number}"
                folders_and_files.make_folder_at(path_to_models_and_data, folder_name)

                train_X, train_y, test_X, test_y = load_OnHW_data(case, dependency, k_fold_number)
                train_X_filtered, train_y_filtered = filter_train_test(train_X, train_y, case)
                test_X_filtered, test_y_filtered = filter_train_test(test_X, test_y, case)

                extracted_features, features_filtered_train, features_filtered_test = ts_extract_feautures(
                    train_X_filtered, train_y_filtered, test_X_filtered)

                path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)
                folder_name = f"{case}_{dependency}_{k_fold_number}"
                folders_and_files.make_folder_at(path_to_models_and_data, folder_name)

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

# Metric learning on OnHW dataset, saves models in same fashion as before
def OnHW_MetricLearn_train_and_save(case, dependency, k_fold_number, ncomp):
    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(
        case, dependency, k_fold_number)

    train_X_features.sort_index(axis=1, inplace=True)

    for k in range(1, 50):
        print(f"[INFO] Training: {case}_{dependency}_{k_fold_number}_NCA_{k}NN_ncomp{ncomp}")
        folder_name = f"{case}_{dependency}_{k_fold_number}_NCA_{k}NN_ncomp{ncomp}"
        path_to_model = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA, folder_name)
        model_name, scaler_name, nca_name = f"nca_model_{k}NN_ncomp{ncomp}", f"nca_scaler_{k}NN_ncomp{ncomp}", f"nca_{k}NN_ncomp{ncomp}"
        model = KNeighborsClassifier(n_neighbors=k)

        scaler = StandardScaler() # NCA requires scaled data
        nca = NeighborhoodComponentsAnalysis(n_components=ncomp, random_state=config.RANDOM_STATE)

        train_X_features_transformed = scaler.fit_transform(train_X_features)
        train_X_features_transformed = pd.DataFrame(train_X_features_transformed)
        nca.fit(train_X_features_transformed, train_y)
        train_X_features_transformed = nca.transform(train_X_features_transformed)
        
        folders_and_files.make_folder_at(os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA), folder_name)
        folders_and_files.save_model(path_to_model, scaler_name, scaler)
        folders_and_files.save_model(path_to_model, nca_name, nca)

        model.fit(train_X_features_transformed, train_y)
        folders_and_files.save_model(path_to_model, model_name, model)



def OnHW_MetricLearn_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, ncomp, k):
    folder_name = f"{case}_{dependency}_{k_fold_number}_NCA_{k}NN_ncomp{ncomp}"
    path_to_model = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA, folder_name)
    model_name, scaler_name, nca_name = f"nca_model_{k}NN_ncomp{ncomp}", f"nca_scaler_{k}NN_ncomp{ncomp}", f"nca_{k}NN_ncomp{ncomp}"
    model = folders_and_files.load_model(path_to_model, model_name)
    scaler = folders_and_files.load_model(path_to_model, scaler_name)
    nca = folders_and_files.load_model(path_to_model, nca_name)

    return model, scaler, nca

def OnHW_MetricLearn_evaluate_model(case, dependency, k_fold_number, ncomp, k):
    model, scaler, nca = OnHW_MetricLearn_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, ncomp, k)

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


def OnHW_MetricLearn_train_all():
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                for ncomp in config.NCOMP_LIST:
                    OnHW_MetricLearn_train_and_save(case, dependency, k_fold_number, ncomp)

def OnHW_MetricLearn_evaluate_all():
    path_to_results = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, config.ML_RESULTS_VARIOUS_NCOMP_CSV)
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                for ncomp in config.NCOMP_LIST:
                    scores_list = []

                    for k in range(1, 50):
                        print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_NCA_{k}NN_ncomp{ncomp}")
                        accuracy, report, conf_mat = OnHW_MetricLearn_evaluate_model(case, dependency, k_fold_number, ncomp, k)
                        L = [[case, dependency, k_fold_number, f"NCA_{k}NN_ncomp{ncomp}", accuracy]]
                        df_results = pd.DataFrame(L, columns=['case', 'dependency', 'fold', 'model', 'accuracy'])
                        if os.path.exists(path_to_results):
                            df_results.to_csv(path_to_results, mode = 'a', index=False, header=False)
                        else:
                            df_results.to_csv(path_to_results, index = False)

                        classification_report_file = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"classification_report_{case}_{dependency}_{k_fold_number}_NCA_{k}NN_ncomp{ncomp}.txt")
                        with open(classification_report_file, 'w') as f:
                            print(report, file=f)

                        conf_mat_csv_file = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"conf_mat_{case}_{dependency}_{k_fold_number}_NCA_{k}NN_ncomp{ncomp}.csv")
                        df_conf_mat = pd.DataFrame(conf_mat)
                        df_conf_mat.to_csv(conf_mat_csv_file, index=False)

                        scores_list.append(accuracy)
                    
                    # Saving accuracy for each nsig as a pickled .txt file
                    score_list_filepath = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"NCA_kNN_fold{k_fold_number}_ncomp{ncomp}.txt")
                    with open(score_list_filepath, 'wb') as fp:
                        pickle.dump(scores_list, fp)



def make_some_folders():
    folders_and_files.make_folder_at('.', config.BASE_OUTPUT)

    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.ML_RESULTS)
    folders_and_files.make_folder_at(os.path.join(config.BASE_OUTPUT, config.ML_RESULTS), config.ML_RESULTS_VARIOUS_NCOMP_CSV)
    folders_and_files.make_folder_at(os.path.join(config.BASE_OUTPUT, config.ML_RESULTS), config.ML_RESULTS_VARIOUS_NSIG_CSV)
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
    make_some_folders()

    '''Run OnHW_ML_filter_and_extract() only once. It takes long time to run through 3*2*5=30 folders
    Each folder is done in about 15 min for lower and upper, 45 min for combined/both ==> 14 hours, in total
    CHANGE XY TO MetricLearn TO RUN ML ALGORITHMS WITH METRIC LEARNING'''

    # OnHW_ML_filter_and_extract()
    OnHW_MetricLearn_train_all()
    OnHW_MetricLearn_evaluate_all()


