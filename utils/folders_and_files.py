import os
import pickle
import numpy as np


# Creates the folder folder_name at path_to_folder: path_to_folder/folder_name
def make_folder_at (path_to_folder, folder_name):
    new_path = os.path.join(path_to_folder, folder_name)
    if not os.path.exists(new_path):
        try:
            os.mkdir(new_path)
        except OSError:
            print('[INFO] Creation of the directory {} failed'.format(new_path))
        else:
            print('[INFO] Successfully created the directory {}'.format(new_path))
    else:
        print('[INFO] Directory {} already exists'.format(new_path))

def save_model(path_to_model, model_name, model):
    file_name = os.path.join(path_to_model, f"{model_name}.pkl")
    if model!=None:
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)

def load_model(path_to_model, model_name):
    file_name = os.path.join(path_to_model, f"{model_name}.pkl")
    try:
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
    except IOError:
        model = None

    return model
