from importlib.resources import path

from utils import config
from utils import folders_and_files

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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC




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

def ts_extract_feautures(train_X, train_y, test_X, nsig):                                                                                                 
    df_train_X = X_npy_to_df(train_X)
    df_test_X = X_npy_to_df(test_X)

    extracted_features = extract_features(df_train_X, column_id='id', column_sort="time")
    # remove all NaN values
    impute(extracted_features)
    features_filtered_train = select_features(extracted_features, train_y, multiclass=True, n_significant=nsig)                           

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
                for nsig in config.NSIG_LIST:
                    print(f"Processing dataset:{case}_{dependency}_{k_fold_number}_nsig{nsig}")                                                                    
                    folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"                                                                                
                    folders_and_files.make_folder_at(path_to_models_and_data, folder_name)

                    train_X, train_y, test_X, test_y = load_OnHW_data(case, dependency, k_fold_number)
                    train_X_filtered, train_y_filtered = filter_train_test(train_X, train_y, case)
                    test_X_filtered, test_y_filtered = filter_train_test(test_X, test_y, case)

                    extracted_features, features_filtered_train, features_filtered_test = ts_extract_feautures(
                        train_X_filtered, train_y_filtered, test_X_filtered, nsig)

                    path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)                                               
                    folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"                                                                                
                    folders_and_files.make_folder_at(path_to_models_and_data, folder_name)

                    np.save(os.path.join(path_to_models_and_data, folder_name, 'train_X_filtered.npy'), train_X_filtered)
                    np.save(os.path.join(path_to_models_and_data, folder_name, 'train_y_filtered.npy'), train_y_filtered)
                    np.save(os.path.join(path_to_models_and_data, folder_name, 'test_X_filtered.npy'), test_X_filtered)
                    np.save(os.path.join(path_to_models_and_data, folder_name, 'test_y_filtered.npy'), test_y_filtered)
                    features_filtered_train.to_csv(
                        os.path.join(path_to_models_and_data, folder_name, 'features_filtered_train.csv'), index=False)
                    features_filtered_test.to_csv(
                        os.path.join(path_to_models_and_data, folder_name, 'features_filtered_test.csv'), index=False)









if __name__ == "__main__":
    # os.chdir('/Volumes/T7') # For accessing feature data (Steven)

    '''For Figure 3 and 4, used all 5 folds of lowercase writer-independent data'''
    # folders_and_files.make_folder_at('.', config.BASE_OUTPUT)
    OnHW_ML_filter_and_extract()
