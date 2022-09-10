import os

RANDOM_STATE = 112358
'''Base folder names'''
BASE_DATASET = 'dataset'
BASE_OUTPUT = 'output'

VISUALS = 'visuals'
ML_MODELS_AND_DATA = 'ml_models_and_data'
ML_RESULTS = 'ml_results'
ML_RESULTS_CSV = 'ml_all_results.csv'
ML_RESULTS_MEAN_CSV = 'ml_results_mean.csv'
ML_RESULTS_FOLD_CSV = 'ml_results_fold.csv'
ML_RESULTS_FIG = "ml_results_summary.png"

'''For generating Figure 3'''
ML_RESULTS_VARIOUS_NSIG_CSV = 'ml_various_nsig_results.csv'
'''For generating Figure 4'''
ML_RESULTS_VARIOUS_NCOMP_CSV = 'ml_various_ncomp_results.csv'

DL_MODELS_AND_DATA = 'dl_models_and_data'
DL_RESULTS = 'dl_results'
DL_RESULTS_CSV = 'dl_all_results.csv'
DL_RESULTS_MEAN_CSV = 'dl_results_mean.csv'
DL_RESULTS_FOLD_CSV = 'dl_results_fold.csv'
DL_RESULTS_FIG = "dl_results_summary.png"

DL_RESULTS_OPT = 'dl_results_optimized'
OPTUNA = 'dl_optuna'
DL_RESULTS_OPT_CSV = 'dl_all_results_opt.csv'
DL_RESULTS_MEAN_OPT_CSV = 'dl_results_mean_opt.csv'
DL_RESULTS_FOLD_OPT_CSV = 'dl_results_fold_opt.csv'
DL_RESULTS_FIG_OPT = "dl_results_summary_opt.png"

'''Prepend for the OnHW dataset'''
PATH_TO_PREPEND = os.path.join(BASE_DATASET, 'IMWUT_OnHW-chars_dataset_2021-06-30', 'OnHW-chars_2021-06-30')

'''There are 3*2*5=30 cases in total and preprocessing and training of ML/DL/DL-optimized/Ensemble methods
take significant time and memory. For simplicity, one can the code over one case: 
lower-independent dataset, fold-0 by using
OnHW_CASE = ['lower']
OnHW_DEPENDENCY = ['indep']
OnHW_FOLD = [0] instead of the following lists specified as below'''
OnHW_CASE = ['lower', 'upper', 'both']
OnHW_DEPENDENCY = ['indep','dep']
OnHW_FOLD = [0,1,2,3,4]

'''The OnHW dataset has 13 = 4*3+1 features'''
NUM_FEATURES = 13

'''Used for filtering, common choices are 1, 2, 3
   Also see above OnHW_CSV_FILE_by_writer_outlier_ids
   (mean - cut_of_coeff*std, mean + cut_of_coeff*std)
'''
CUT_OFF_COEFF = 2

'''Bounds for filtering ts data wrt the number of time steps'''
HARD_LOWER_BOUND, HARD_UPPER_BOUND = 10, 200

'''mean and std of lower-case, upper-case, and combined characters in the OnHW dataset'''
MEAN = {'lower': 44.05, 'upper': 52.85, 'both': 48.45}
STD = {'lower': 29.93, 'upper': 42.82, 'both': 37.20}

'''Padding timeseries to a foxed-length of MAXLEN = 104 time steps'''
MAXLEN=104

'''Optimal parameter (wrt kNN) for tsfresh when extracting features'''
NUM_SIG=17
N_COMPONENTS = 21

'''For creating Figure 3 and Figure 4 in the paper'''
NSIG_LIST = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
NCOMP_LIST = [8, 12, 16, 20, 21, 22, 24, 32, 64, 128, 256]

'''A default value for epoch to be used in tsai archs'''
TSAI_EPOCH = 50

'''Change this parameter to optimize other models over other datasets
The code currently allows the following models to be optimized:
InceptionTimePlus, LSTM, LSTM_FCN and MLSTM_FCN (see "OnHW_DL_optimize.py" for more details)'''
TO_OPTIMIZE = ['InceptionTimePlus', 'lower', 'indep', 0]