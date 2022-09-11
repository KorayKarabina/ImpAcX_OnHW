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
from sklearn import metrics


# Produces Figure 3, comparing the performance of KNN over various levels of n-significant



def OnHW_ML_read_filtered_data_and_extracted_features(case, dependency, k_fold_number, nsig):
    path_to_models_and_data = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA)                                                           
    folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"

    train_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_X_filtered.npy"), allow_pickle=True)
    train_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "train_y_filtered.npy"), allow_pickle=True)
    test_X_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_X_filtered.npy"), allow_pickle=True)
    test_y_filtered = np.load(os.path.join(path_to_models_and_data, folder_name, "test_y_filtered.npy"), allow_pickle=True)
    features_filtered_train = pd.read_csv(os.path.join(path_to_models_and_data, folder_name, "features_filtered_train.csv"))
    features_filtered_test = pd.read_csv(os.path.join(path_to_models_and_data, folder_name, "features_filtered_test.csv"))

    return train_X_filtered, train_y_filtered, test_X_filtered, test_y_filtered, features_filtered_train, features_filtered_test

def OnHW_ML_train_and_save(case, dependency, k_fold_number, nsig):                                                                 
    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(                        
        case, dependency, k_fold_number, nsig)

    for k in range(1,50):
        print(f"[INFO] Training: {case}_{dependency}_{k_fold_number}_{k}NN")
        folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"                                                                                        
        path_to_model = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA, folder_name)                                                    
        model_name, scaler_name = f"model_{k}NN_nsig{nsig}", f"scaler_{k}NN_nsig{nsig}"
        model, scaler = KNeighborsClassifier(n_neighbors=k), QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=config.RANDOM_STATE)

        train_X_features_transformed = scaler.fit_transform(train_X_features)
        folders_and_files.save_model(path_to_model, scaler_name, scaler)

        model.fit(train_X_features_transformed, train_y)
        folders_and_files.save_model(path_to_model, model_name, model)




def OnHW_ML_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, nsig, k):
    folder_name = f"{case}_{dependency}_{k_fold_number}_nsig{nsig}"                                                                                
    path_to_model = os.path.join(config.BASE_OUTPUT, config.ML_MODELS_AND_DATA, folder_name)                                            
    model_name, scaler_name = f"model_{k}NN_nsig{nsig}", f"scaler_{k}NN_nsig{nsig}"
    model = folders_and_files.load_model(path_to_model, model_name)
    scaler = folders_and_files.load_model(path_to_model, scaler_name)

    return model, scaler



def OnHW_ML_evaluate_model(case, dependency, k_fold_number, nsig, k):                                                                       
    model, scaler = OnHW_ML_load_a_model_and_scaler_by_name(case, dependency, k_fold_number, nsig, k)                                   

    train_X, train_y, test_X, test_y, train_X_features, test_X_features = OnHW_ML_read_filtered_data_and_extracted_features(            
        case, dependency, k_fold_number, nsig)

    if scaler==None:
        test_X_features_transformed = test_X_features.copy()
    else:
        test_X_features_transformed = scaler.transform(test_X_features)

    preds = model.predict(test_X_features_transformed)
    report = classification_report(test_y, preds, zero_division=1, digits=4)
    conf_mat = confusion_matrix(test_y, preds)
    accuracy = accuracy_score(test_y, preds)

    return accuracy, report, conf_mat



def OnHW_ML_train_all():
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:
                for nsig in config.NSIG_LIST:                                                                                      
                    OnHW_ML_train_and_save(case, dependency, k_fold_number, nsig)                                       

               

def OnHW_ML_evaluate_all():
    path_to_results = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, config.ML_RESULTS_VARIOUS_NSIG_CSV)                                        
    for case in config.OnHW_CASE:
        for dependency in config.OnHW_DEPENDENCY:
            for k_fold_number in config.OnHW_FOLD:                                                                                      
                for nsig in config.NSIG_LIST:
                    # Score list holds the score for each level of nsig
                    scores_list = []

                    for k in range(1, 50):                                                                                                   
                        print(f"[INFO] Evaluating: {case}_{dependency}_{k_fold_number}_{k}NN_nsig{nsig}")
                        accuracy, report, conf_mat = OnHW_ML_evaluate_model(case, dependency, k_fold_number, nsig, k)                          
                        L = [[case, dependency, k_fold_number, f"{k}NN_nsig{nsig}", accuracy]]                                                 
                        df_results = pd.DataFrame(L, columns=['case', 'dependency', 'fold', 'model', 'accuracy'])                           
                        if os.path.exists(path_to_results):
                            df_results.to_csv(path_to_results, mode = 'a', index=False, header=False)
                        else:
                            df_results.to_csv(path_to_results, index = False)

                        classification_report_file = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"classification_report_{case}_{dependency}_{k_fold_number}_{k}NN_nsig{nsig}.txt")
                        with open(classification_report_file, 'w') as f:
                            print(report, file=f)

                        conf_mat_csv_file = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"conf_mat_{case}_{dependency}_{k_fold_number}_{k}NN_nsig{nsig}.csv")                             
                        df_conf_mat = pd.DataFrame(conf_mat)
                        df_conf_mat.to_csv(conf_mat_csv_file, index=False)

                        scores_list.append(accuracy)

                    # Saving accuracy for each nsig as a pickled .txt file
                    score_list_filepath = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"kNN_fold{k_fold_number}_nsig{nsig}.txt")
                    with open(score_list_filepath, 'wb') as fp:
                        pickle.dump(scores_list, fp)






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
    make_some_folders()

    '''Run OnHW_ML_extract_features_various_nsig.py prior to extract features for all levels of n_significant, specified in config.NSIG_LIST
    CHANGE XY TO ML TO RUN ML ALGORITHMS
    CHANGE XY TO MetricLearn TO RUN ML ALGORITHMS WITH METRIC LEARNING'''

    OnHW_ML_train_all()
    OnHW_ML_evaluate_all()


