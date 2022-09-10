from importlib.resources import path
from string import ascii_letters, ascii_lowercase, ascii_uppercase

import numpy as np
from utils import config
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tslearn import svm,utils
import os
from re import I
from dtaidistance import dtw_ndim
from matplotlib import pyplot as plt
import shutil
import pytwed
from collections import Counter
import string
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn import svm
from tslearn.metrics import soft_dtw
i=0

def fit_SVM(kernel:str, X, Y):
    if kernel=="DTW":
        clf = svm.SVC(kernel='precomputed')
    clf.fit(get_gram(X, X, DTW3), Y)
    return clf

def get_gram(x1, x2, kernel):
    return np.array([[kernel(_x1, _x2) for _x2 in x2] for _x1 in x1])



# Matrix manipulation helpers for single matrices
def stand(mat: np.array):
    """Standardizes matrix. (x-mean)/stdev

    Args:
        mat ([np.array]): matrix to be standardized
        type (str, optional): Specified type of standardization. Defaults to "Normal".

    Returns:
        [type]: [description]
    """
    return (mat - np.mean(mat, axis=0)) / np.std(mat, axis=0)


def diff(mat: np.array):
    """Returns a matrix of the difference of consecutive elements for each column

    Args:
        mat (np.array): matrix

    Returns:
        np.array: matrix
    """
    return np.diff(mat, axis=0)


def norm(mat: np.array):
    """Returns normalized array

    Args:
        mat (np.array): matrix

    Returns:
        np.array: normalized matrix
    """
    norm_a = np.linalg.norm(mat, axis=0)
    return mat/norm_a


def log(mat: np.array):
    """returns log of array

    Args:
        mat (np.array): matrix

    Returns:
        np.array: log array
    """
    return np.log(mat)


def minmax(mat: pd.DataFrame, norm: MinMaxScaler):
    """Returns the minmax norm of the array

    Args:
        mat (pd.DataFrame): array in dataframe format
        norm (MinMaxScaler): instance of sklearns MinMaxScaler

    Returns:
        pd.DataFrame: minmax norm of the array
    """
    return norm.fit_transform(mat)


def preprocess_data(x: np.array, y: np.array, type: str, min_length: int, max_length: int):
    """Runs through each element in the data.
    Removes elements that are too short or too long
    manipulates the relevent data according to the specified type

        -   "minmax" applies sklearn's minmax normalization
        -   "standscale" applies sklearn's robustscaler standardization
        -   "stand" for standardization with mean and standard deviation
        -   "raw" to change nothing in the array
        -   "diff" to return array with differences of conecutive elements
        -   "norm" divide each element by the frobenius norm of the column
        -   "log" take the log of each element 

        more than one of these can be applied, seperated by "_"

    Args:
        x (np.array): array of X_train/X_test, the sensor data 
        y (np.array): [label data]
        type (str): pre-processing type
        min_length (int): delete entries shorter than this length
        max_length (int): delete entries longer than this length

    Returns:
        [np.array]: x, but preprocessed
    """

    norm_sc = MinMaxScaler()
    RobustScaler = QuantileTransformer()
    length = len(x)-1
    i = 0
    # loop through each element
    while i < length:

        df = pd.DataFrame(x[i])
        if (len(df) < min_length)or (len(df)>max_length):
            x = np.delete(x, i, 0)
            y = np.delete(y, i, 0)
            i = i-1
            length = length - 1
            continue

        if "minmax" == type:
            df_scaled = pd.DataFrame(norm_sc.fit_transform(df))
            x[i] = df_scaled.to_numpy()
        elif "standscale" == type:
            df_scaled = pd.DataFrame(RobustScaler.fit_transform(df))
            x[i] = df_scaled.to_numpy()
        else:
            x[i] = df.to_numpy()
            x[i] = matrix_manipulation(x[i], type, norm_sc, RobustScaler)
        i = i+1
    return x, y

# Matrix manipulation of a, according to type specified
def add_multiclass(y:np.array,multiclass_mapping:dict):
    for i, label in enumerate(y):
        y[i] = multiclass_mapping[label]
    return y

def to_multiclass(matches:list,multiclass_mapping):
    new_matches = []
    for m in matches:
        new_matches.append((multiclass_mapping[m[0]],multiclass_mapping[m[1]]))
    return new_matches


def matrix_manipulation(a: np.array, type: str, norm_sc: MinMaxScaler = None, robust_sc: RobustScaler = None):
    """manipulate matrix as specified
        -   "minmax" applies sklearn's minmax normalization
        -   "standscale" applies sklearn's robustscaler standardization
        -   "stand" for standardization with mean and standard deviation
        -   "raw" to change nothing in the array
        -   "diff" to return array with differences of conecutive elements
        -   "norm" divide each element by the frobenius norm of the column
        -   "log" take the log of each element 
        more than one of these can be applied, seperated by "_"

    Args:
        a (np.array): matrix to be manipulated as specified
        type (str): type of manipulation to apply
        norm_sc (MinMaxScaler, optional): SKLearnsMinMaxScaler if needed for manipulation. Defaults to None.
        robust_sc (RobustScaler, optional): SKLearns RobustScaler if needed for manipulation. Defaults to None.

    Returns:
        np.array: manipulated array
    """

    type_split = type.split("_")
    for t in type_split:
        if t == "raw":
            pass
        elif t == "stand":
            a = stand(a)
        elif t == "stand2":
            a = stand(a, "robust")
        elif t == "diff":
            a = diff(a)
        elif t == "norm":
            a = norm(a)
        elif t == "log":
            a = log(a)
        elif t == "minmax":
            df = pd.DataFrame(a)
            df_scaled = pd.DataFrame(norm_sc.fit_transform(df))
            a = df_scaled.to_numpy()
        elif t == "standscale":
            df = pd.DataFrame(a)
            df_scaled = pd.DataFrame(robust_sc.fit_transform(df))
            a = df_scaled.to_numpy()
    return a

# Calculates DTW


def DTW(a: np.array, b: np.array):
    """calculates the distance between two arrays, using DTW

    Args:
        a (np.array): first array
        b (np.array): second array

    Returns:
        int: distance
    """
    global i
    i=i+1
    x = dtw_ndim.distance_fast(a, b)
    return x


def DTW2(a: np.array, b: np.array):
    """calculates the distance between two arrays, using DTW

    Args:
        a (np.array): first array
        b (np.array): second array

    Returns:
        int: distance
    """
    global i
    i=i+1
    x = soft_dtw(a, b, gamma=1)
    return np.exp(-x)

def DTW3(a: np.array, b: np.array):
    """calculates the distance between two arrays, using DTW

    Args:
        a (np.array): first array
        b (np.array): second array

    Returns:
        int: distance
    """
    global i
    i=i+1
    x = dtw_ndim.distance_fast(a, b)
    return np.exp(-x)



# Calculates TWED, assumes first column is time


def TWED(a: np.array, b: np.array, a_ts: np.array, b_ts: np.array):
    """calculates the distance between two arrays, using the TWED metric

    Args:
        a (np.array): first array
        b (np.array): second array
        a_ts (np.array): time series for first array
        b_ts (np.array): time series for second array

    Returns:
        int: distance
    """
    return pytwed.twed(a, b, a_ts, b_ts, nu=0.001, lmbda=1.0, p=2, fast=True)


def TWED(a: np.array, b: np.array):
    """calculates the distance between two arrays, using the TWED metric

    Args:
        a (np.array): first array
        b (np.array): second array
        a_ts (np.array): time series for first array
        b_ts (np.array): time series for second array

    Returns:
        int: distance
    """
    a_ts = np.arange(a.shape[0])
    b_ts = np.arange(b.shape[0])
    return pytwed.twed(a, b, a_ts, b_ts, nu=0.001, lmbda=1.0, p=2, fast=True)


# get distances between train and test return a tuple of distances and labels, sorted by distance
def get_neighbors(train: np.array, train_labels: np.array, test_row: np.array, distance_metric: str = "DTW"):
    """for the row of the test data specified, calculate the distance from the point in the test to each point in the training set using the distance metric specified
        returns list of tuple in order of distance, as distance, label, where label is the label of the train data corresponding to the distance

    Args:
        train (np.array): training set
        train_labels (np.array): training labels
        test_row (np.array): single point in the testing data
        distance_metric (str, optional): Distance metric. Either "DTW" or "TWED". Defaults to "DTW".

    Returns:
        list: list of tuples of distance, label
    """
    distances = list()
    i = 0
    for train_row in train:
        # loop through each element
        a, b = train_row.copy(), test_row.copy()
        if distance_metric == "DTW":
            try:
                dist = DTW(a, b)
            except IndexError:
                dist = 100000
        elif distance_metric == "TWED":
            try:
                a_ts = np.arange(a.shape[0])
                b_ts = np.arange(b.shape[0])
                dist = TWED(a, b)
            except IndexError:
                dist = 100000

        distances.append((dist, train_labels[i]))
        i = i+1
    # sort distances from
    distances.sort(key=lambda tup: tup[0])
    return distances

# take in k nearest neighbors and return most common, if theres a tie, try again with k-1


def get_max(neighbors:list):
    """get most common label in the group
    Recursive function which will find the most common in the group minus the last element if there is a tie

    Args:
        neighbors (list): list of neighbors

    Returns:
        str: maximum label
    """   
    c = Counter(neighbors)
    maximums = [x for x in c if c[x] == c.most_common(1)[0][1]]
    if len(maximums) != 1:
        return get_max(neighbors[:-1])
    else:
        return maximums[0]


# Run KNN model
def knn(train:np.array, train_labels:np.array, test:np.array, test_labels:np.array, num_neighbours:int,):
    """for each element in the test dataset, make a prediction using knn as to what the label should be. return a list of tuples with the predicted labels and actual labels

    Args:
        train (np.array): train dataset
        train_labels (np.array): training labels
        test (np.array): test dataset
        test_labels (np.array): testing labels
        num_neighbours (int): k in k nearest neighbors

    Returns:
        list: list of tuples with predicted label, actual label
    """    
    i = 0
    output_matches = list()
    for row in test:
        p = predict_classification(
            train, train_labels, row, num_neighbours)
        output_matches.append((p, test_labels[i]))
        i = i+1
    return output_matches
# Score the model


def score_knn(output_matches:list):
    """get the accuracy of the predicted data

    Args:
        output_matches (list): list of tuples of predicted label, real label

    Returns:
        float: percentage accuracy in terms of float
    """    
    total = len(output_matches)
    count = 0
    for i in output_matches:
        if i[0] == i[1]:
            count = count+1
    return count/total
# Create confusion matrix


def create_confusion_matrix(output_matches:list,classes:list,title = None,save = False):
    """Creates a confusion matrix for the list of predicted labels vs actual labels

    Args:
        output_matches (list): list of tuples of predicted label, real label
    """    

    y_true = list()
    y_pred = list()
    for i in output_matches:
        y_true.append(i[1])
        y_pred.append(i[0])
    cm = confusion_matrix(
            y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classes)


    disp.plot()
    if save:
        fig, ax = plt.subplots(figsize=(10,10))
        disp.plot(ax=ax)
        plt.savefig('confusion_matrix_ '+ title+".png")
        plt.clf()
    else:
        plt.show()

def letter_scores_knn(output_matches:list):
    """Score each letter individually and return a dictionary with the accuracy of each letter seperately 

    Args:
        output_matches (list): list of tuples of predicted label, real label

    Returns:
        dictionary: dictionay for each letter as the key and the accuracy as value
    """    

    try:
        d = dict.fromkeys(string.ascii_lowercase, 0)
        d_total = dict.fromkeys(string.ascii_lowercase, 0)

        for i in output_matches:
            d_total[i[1]] = d_total[i[1]] + 1
            if i[0] == i[1]:
                d[i[1]] = d[i[1]] + 1
        for i in d_total.keys():
            d[i] = "{:.2%}".format(d[i]/d_total[i])
    except:
        d = dict.fromkeys(string.ascii_uppercase, 0)
        d_total = dict.fromkeys(string.ascii_uppercase, 0)
        for i in output_matches:
            d_total[i[1]] = d_total[i[1]] + 1
            if i[0] == i[1]:
                d[i[1]] = d[i[1]] + 1
        for i in d_total.keys():
            d[i] = "{:.2%}".format(d[i]/d_total[i])
    return d


def predict_classification(train:np.array, train_labels:np.array, test_row:np.array, num_neighbors:int):
    """Make prediction based for a test element, given the training data, using KNN

    Args:
        train (np.array): train dataset
        train_labels (np.array): training labels
        test_row (np.array): test dataset
        num_neighbors (int): k in knn

    Returns:
        [type]: prediction label
    """     
    distances = get_neighbors(train, train_labels, test_row)
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i])
    output_values = [row[-1] for row in neighbors]
    # prediction = max(set(output_values), key=output_values.count)
    prediction = get_max(output_values)
    return prediction


def predict_all(distances:list, num_neighbors:int):
    """give prediction given all the distances

    Args:
        distances (list): distances between test and each train row
        num_neighbors (int): k in knn

    Returns:
        str: predicted label
    """   
    distances.sort(key=lambda tup: tup[0])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i])
    output_values = [row[-1] for row in neighbors]
    prediction = get_max(output_values)
    return prediction

def get_class_report(output_matches:list,classes):
    """return sklearns classification report

    Args:
        output_matches (list): list of tuples of predicted label, real label

    Returns:
        dict: classification report
    """    
    y_true = list()
    y_pred = list()
    for i in output_matches:
        y_true.append(i[1])
        y_pred.append(i[0])
    #try with display with lowercase, if that doesnt work, try upper case
    cf = classification_report(y_true, y_pred, target_names=list(classes))
    return cf

def knn_from_neighbors(neighbors:list, labels:np.array, num_neighbors:int):
    """make prediction given all of the neighbours

    Args:
        neighbors (list): list of distances and neighbors between each row in train with the test row
        labels (np.array): actual labels
        num_neighbors (int): k in knn

    Returns:
        str: prediction
    """    

    i = 0
    matches = list()
    for row in neighbors:
        p = predict_all(row, num_neighbors)
        matches.append((p, labels[i]))
        i = i+1
    return matches


def all_neighbors(train:np.array, train_labels:np.array, test:np.array, distance_metric:int,amt_neighbors = -1,save = False,path = None):
    """returns distances between each element in test and each element in train

    Args:
        train (np.array): training 
        train_labels (np.array): training labels
        test (np.array): testing data
        distance_metric (int): which distance metric is used for distances

    Returns:
        list: list of all distances and labels
    """
    if save and (not os.path.exists(path)):
        create_file(path)
    i = -1
    all_neighbors = list()
    for row in test:
        i = i+1
        if save and os.path.exists(path+"/"+str(i)+".csv"):
            continue
        neighbors = get_neighbors(
            train, train_labels, row, distance_metric)
        all_neighbors.append(neighbors[0:amt_neighbors])
        if save:
            neighbor_to_csv(i,neighbors[0:amt_neighbors],path)
    return all_neighbors


def neighbors_to_csv(neighbors:list, path:str):
    """write the neighbours to many csv files

    Args:
        neighbors (list): list of lists of tuples where each tuple inner list corresponds to a test letter and each tuple corresponds to the distance to a training element, along with said label
        path (str): path to write files to
    """    
    i = 0
    create_file(path)

    for n in neighbors:
        neighbor_to_csv(i,n,path)
        i = i + 1

def neighbor_to_csv(i,neighbor:list, path:str):
    """write the neighbours to many csv files

    Args:
        neighbors (list): list of lists of tuples where each tuple inner list corresponds to a test letter and each tuple corresponds to the distance to a training element, along with said label
        path (str): path to write files to
    """    
    df = pd.DataFrame(neighbor, columns=['distance', "label"])
    df.to_csv(path+"/"+str(i)+".csv", index=False)
    del df

def create_file(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def read_neighbors_from_csv(path):
    """read neigbours to list of list of tuples

    Args:
        path (str): path to read from

    Returns:
        list: list of lists of tuples where each tuple inner list corresponds to a test letter and each tuple corresponds to the distance to a training element, along with said label
    """    
    neighbors = []
    files = os.listdir(path)
    files_2 = [int(i[:-4]) for i in files]
    files_2.sort()

    for file in files_2:
        df = pd.read_csv(path+"/"+str(file)+".csv")
        neighbors.append(df.values.tolist())
        del df
    return neighbors


# Uses neighbors to create k-graph
def create_k_graph(neighbors, labels, title, num_neighbors_range=(1, 60),save = False):
    """Create k-graph 

    Args:
        neighbors (list): list of lists of tuples where each tuple inner list corresponds to a test letter and each tuple corresponds to the distance to a training element, along with said label
        labels (list): true labels of test   
        title (str): title of graph
        num_neighbors_range (tuple, optional): range of k in knn to try. Defaults to (1, 60).
    """    
    scores = []
    k_s = []
    for k in range(num_neighbors_range[0], num_neighbors_range[1]):
        matches = knn_from_neighbors(neighbors, labels, k)
        k_s.append(k)
        scores.append(score_knn(matches))
    plt.plot(k_s, scores)
    plt.title('K-graph '+title)
    plt.xlabel('K')
    plt.ylabel('scores')
    if save:
        plt.savefig('K_graph_ '+ title+".png")
        plt.clf()
    else:
        plt.show()


def results_to_csv(data,path):

    df = pd.DataFrame(data, columns=['case', "dependency",'fold','model','accuracy'])
    df.to_csv(path+".csv", index=False)


# Identifier for the version of data

# Distance metric to use. Currently supported: "DTW", "TWED"
DISTANCE_METRIC = "DTW"
multiclass_version = 0



def run_dtw_knn(model,TYPE):
    scores = []
    class_reports = []


    data_to_csv = []


    # Loop through each case
    for case in config.OnHW_CASE:
        # Loop through dependencies WI vs WD
        for dependency in config.OnHW_DEPENDENCY:
            # Loop through each fold in the identifier
            for fold in config.OnHW_FOLD:
                identifier = "onhw2_"+case+"_"+dependency+ "_"+fold
                if "both" in identifier:
                    classes = list(ascii_letters)
                elif "upper" in identifier:
                    classes = list(ascii_uppercase)
                else:
                    classes = list(ascii_lowercase)
                path = identifier+"_"+TYPE+"_"+DISTANCE_METRIC#+"_mc"+str(multiclass_version)
                #Load in data
                X_train = np.load('onhw-chars_2021-06-30/'+identifier +
                                '/X_train.npy', allow_pickle=True)
                y_train = np.load('onhw-chars_2021-06-30/'+identifier +
                                '/y_train.npy', allow_pickle=True)

                X_test = np.load('onhw-chars_2021-06-30/'+identifier +
                                '/X_test.npy', allow_pickle=True)
                y_test = np.load('onhw-chars_2021-06-30/'+identifier +
                                '/y_test.npy', allow_pickle=True)


                # Manipulate the train and test data according to specified preprocessing techniques above
                X_train, y_train = preprocess_data(X_train, y_train, TYPE, config.HARD_LOWER_BOUND, config.MAXLEN)

                X_test, y_test = preprocess_data(X_test, y_test, TYPE, config.HARD_LOWER_BOUND, config.MAXLEN)

                # The following 4 lines  can be commented out if the neighbors have already been calculated and stored for the relevant fold and preprocessing
                if model == "KNN":
                    neighbors = all_neighbors(X_train.copy(),y_train.copy(), X_test.copy(),DISTANCE_METRIC,amt_neighbors=200,save = True,path=path)
                    print("DONE GETTING NEIGHBORS")
                    # Read in the neighbors
                    try:
                        n = read_neighbors_from_csv(path)
                    except Exception as e:
                        print("Try uncommenting above lines in code")
                        print(e)
                        quit()
                    print("DONE READING")
                    # Calculate the matches
                    matches = knn_from_neighbors(n, y_test, 5)

                    # Create K-graph
                    # create_k_graph(n, y_test, path,save = True)


                    score = score_knn(matches)
                    scores.append(score)
                    # class_reports.append(get_class_report(matches,classes))
                    data_to_csv.append({'case':case, "dependency":dependency,'fold':fold,'model':"KNN_"+DISTANCE_METRIC,'accuracy':score})
                elif model == "SVM":

                    clf = fit_SVM(DISTANCE_METRIC,X_train,y_train)
                    print(f'Accuracy on Custom Kernel: {accuracy_score(y_test, clf.predict(get_gram(X_test, X_train, DTW3)))}')
                elif model == "KNN_tslearn":
                    for i, x in enumerate(X_train):
                        X_train[i] = utils.to_time_series(np.concatenate((np.arange(x.shape[0])[:, np.newaxis], x), axis=1))
                    for i, x in enumerate(X_test):
                        X_test[i] = utils.to_time_series(np.concatenate((np.arange(x.shape[0])[:, np.newaxis], x), axis=1))
                    X_train_dataset = utils.to_time_series_dataset(X_train)
                    X_test_dataset = utils.to_time_series_dataset(X_test)
                    print("Done step 1")

                    # clf = svm.TimeSeriesSVC(C=1.0, kernel="gak")
                    clf = neighbors.KNeighborsTimeSeriesClassifier()
                    print("Done step 2")
                    fitted_clf = clf.fit(X_train_dataset,y_train)
                    print("Done step 3")
                    predicted = fitted_clf.predict(X_test_dataset)
                    i=0
                    matches = []
                    for i in range(y_test):
                        matches.append((predicted[i],y_test[i]))
                    score = score_knn(matches)
                    print(score)
                    scores.append(score)
                    class_reports.append(get_class_report(matches,classes))

    results_to_csv(data_to_csv,"output_accuracy")
    for c in class_reports:
        print(c)
    print(scores)



if __name__ == "__main__":

    #Code included for KNN_tslearn and SVM, however, the code in its final stage is only maintained for KNN
    model  = "KNN"

    # Type of preprocessing to be applied. "norm" divides each element by the frobenius norm of the column
    # "diff" takes the difference between consecutive elements
    # "minmax" normalizes by the minimum and maximum of the column
    # "stand" standardizes each element by the column
    # "stand2" standardizes using the median, 3rd and 1st quartile
    TYPE = "minmax"

    #Run KNN with DTW
    run_dtw_knn(model,TYPE)
