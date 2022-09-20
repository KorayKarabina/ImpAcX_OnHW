# ImpAcX_OnHW: Improving Accuracy and Explainability of Handwriting Recognition Models

This project contains codes and information to help reproduce, verify and improve results as reported in our paper [Improving Accuracy and Explainability of Handwriting Recognition Models](https://arxiv.org/abs/2209.09102).

## How to cite this project?

If you use this project in your research or development, please cite it as

```
@misc{impacx_onhw,
  author = {H. Azimi and S. Chang and J. Gold and K. Karabina},
  title = {{Improving Accuracy and Explainability of Handwriting Recognition Models}},
  year = {2022},
  publisher = {arXiv},
  pages = {1-20},
  url = {https://arxiv.org/abs/2209.09102},
  doi = {10.48550/ARXIV.2209.09102},
  copyright = {Creative Commons Attribution 4.0 International}
}

```


***

## Description

The project directory looks like

````
├── dataset
├── DTW_KNN.py
├── ensemble_evaluation.py
├── explain.py
├── failure_space_analysis.py
├── failure_space_array.py
├── OnHW_DL_optimize.py
├── OnHW_ML_DL_baseline_and_optimized_train_and_eval.py
├── OnHW_ML_extract_features_various_nsig.py
├── OnHW_ML_train_KNN_various_ncomp.py
├── OnHW_ML_train_KNN_various_nsig.py
├── plot_kNN_results.py
├── README.md
├── requirements.txt
├── results_to_visual.py
└── utils

````

- ````requirements.txt```` lists all the dependencies for this project.

- Before running python codes, the [OnHW-chars dataset](https://stabilodigital.com/onhw-dataset/) should be unzipped and placed under the 'dataset' folder so it looks like

````
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
````

- After the OnHW-chars dataset is placed under the 'dataset' folder as described above, the following python codes can be run for training and evaluating models, and for visualizing results:

  * **Running "OnHW_ML_DL_baseline_and_optimized_train_and_eval.py":** First, a call to ````make_some_folders()```` is made that creates the "output" directory together with several subdirectories under which processed datasets, trained models, transformers, evaluations, results, visualizations, etc. can be saved. The ````make_some_folders()```` line is followed by <br/>

    ````
    OnHW_ML_filter_and_extract()
    OnHW_ML_train_all()
    OnHW_ML_evaluate_all()
    ````
    <br/>
    
    that runs the 
    [tsfresh](https://tsfresh.readthedocs.io/en/latest/) feature extraction algorithm and trains and evaluates 7 machine learning baseline models. Then the same 7 machine learning baseline models are again trained and evaluated using a metric learning transformation <br/>

    ````
    OnHW_MetricLearn_train_all()
    OnHW_MetricLearn_evaluate_all()
    ````
    <br/>

    Then 12 deep learning baseline models are trained and evaluated through <br/>

    ````
    OnHW_DL_train_all()
    OnHW_DL_evaluate_all()
    ````
    <br/>

    Finally, 4 deep learning models (whose parameters are optimized via "OnHW_DL_optimize.py") are trained and evaluated through <br/>
    
    ````
    OnHW_DL_optimized_train_all() 
    OnHW_DL_optimized_evaluate_all()    
    ````
    <br/>

  * **Running "OnHW_DL_optimize.py":** [Optuna](https://optuna.org/) framework is used to optimize 4 deep learning models. Optimized models and their parameter sets can be passed on to the  "DL_OPT_MODEL_LIST" variable in "OnHW_ML_DL_baseline_and_optimized_train_and_eval.py" so that model trainings and evaluations can be performed via <br/>
  
    ````
    OnHW_DL_optimized_train_all() 
    OnHW_DL_optimized_evaluate_all()    
    ````
     <br/>
     
    in "OnHW_ML_DL_baseline_and_optimized_train_and_eval.py". <br/>

  * **Running "OnHW_ML_extract_features_various_nsig.py":** This code will extract and select features for various levels of n_significant to produce Figure 3. In our Figure we used all 5 folds of the lowercase writer-independent data to produce our results. The levels n_significant and the datasets to use are modifiable in "./utils/config.py". </br>

  * **Running "OnHW_ML_train_KNN_various_nsig.py":** After running "OnHW_ML_extract_features_various_nsig.py", this code will train and evaluate KNN models with levels of k ranging from 1 to 49 for various levels of n_significant. The results and accuracy of all these models is saved under "./output/ml_results" will be used to produce Figure 3. </br>

  * **Running "OnHW_ML_train_KNN_various_ncomp.py":** After extracting features will train and evaluate KNN models with levels of k ranging from 1 to 49 for various levels of n_components to produce Figure 4. The parameter n_significant here is chosen to be 17, the optimal level chosen from our visual analysis. In our Figure we used all 5 folds of lowercase writer-independent data, the datasets to use and the specified levels of n_components is modifiable in "./utils/config.py". The results and accuracy of these models is saved under "./output/ml_results", and will used to produce Figure 4. </br>

  * **Running "results_to_visual.py"** After model trainings and evaluations are completed, this code computes per-fold and mean accuracies, standard deviations, etc. over the test dataset and provides a visualisation of results via "ml_dl_all_results.html" that is saved under "./output/visuals". </br>
  
  * **Running "plot_kNN_results.py":** After OnHW_ML_train_KNN_various_ncomp, and OnHW_train_KNN_various_nsig, this code produces Figures 3 and 4 using pyplot and the results saved under "./output/ml_results". The resulting plots are saved under "./output/visuals". </br>

  * **Running "ensemble_evaluation.py"** After model trainings and evaluations are completed, this code evaluates per fold test results for each model specified. The code will produce a classification report, confusion matrix as well as testing accuracy. The ensemble method to be used is expressed as a parameter. The options, as described in the file itself are for weighted soft voting, soft voting, weighted voting, majority voting and lexogrpahic ordering. </br>
  
  * **Running "explain.py"** After model trainings and evaluations are completed, this code produces per fold explanations of each of the models for a specified element of the testing data. This element is specified as a parameter. The code produces two images. The first is a bar chart where the most important channel and time combinations are specified. The second is a graph of the multivariate time series with colours superimposed, corresponding to the important sections of the time series. </br>
  
  * **Running "failure_space_analysis.py"** After model trainings and evaluations are completed, this code produces per fold analysis of the failure spaces of each of the models specified. A bar graph will be produced for each of the models specified, showing the proportion of failures of each other model that overlap. </br>

  * **Running "failure_space_array.py"** After model trainings and evaluations are completed, this code produces per fold analysis of the failure spaces of each of the models specified. A matrix will be produced where each of the models specified correspond to a row and each piece of test data is a column. An element will be white if it is predicted correctly and black if predicted incorrectly. </br>
  
  
## Notes
Running the python codes as describeed above may require significant time and memory. In particular, a successful run of the above seqeunce would result in an "output" directory that looks like

````
├── dl_models_and_data
├── dl_optuna
├── dl_results
├── dl_results_optimized
├── ml_models_and_data
├── ml_results
└── visuals
````

where in particular, "./output/ml_models_and_data/lower_indep_0" would look like

````
├── features_filtered_test.csv
├── features_filtered_train.csv
├── model_DT.pkl
├── model_ET.pkl
├── model_kNN.pkl
├── model_LogReg.pkl
├── model_RFC.pkl
├── model_SVM_linear.pkl
├── model_SVM_rbf.pkl
├── nca_model_metriclearn_DT.pkl
├── nca_model_metriclearn_ET.pkl
├── nca_model_metriclearn_kNN.pkl
├── nca_model_metriclearn_LogReg.pkl
├── nca_model_metriclearn_RFC.pkl
├── nca_model_metriclearn_SVM_linear.pkl
├── nca_model_metriclearn_SVM_rbf.pkl
├── nca_scaler_metriclearn_DT.pkl
├── nca_scaler_metriclearn_ET.pkl
├── nca_scaler_metriclearn_kNN.pkl
├── nca_scaler_metriclearn_LogReg.pkl
├── nca_scaler_metriclearn_RFC.pkl
├── nca_scaler_metriclearn_SVM_linear.pkl
├── nca_scaler_metriclearn_SVM_rbf.pkl
├── nca_scaler_metriclearn_DT.pkl
├── nca_metriclearn_ET.pkl
├── nca_metriclearn_kNN.pkl
├── nca_metriclearn_LogReg.pkl
├── nca_metriclearn_RFC.pkl
├── nca_metriclearn_SVM_linear.pkl
├── nca_metriclearn_SVM_rbf.pkl
├── scaler_kNN.pkl
├── scaler_LogReg.pkl
├── scaler_SVM_linear.pkl
├── scaler_SVM_rbf.pkl
├── test_X_filtered.npy
├── test_y_filtered.npy
├── train_X_filtered.npy
└── train_y_filtered.npy

````

For completeness, we provide our version of the "output" that we refer in our paper here: [output](https://drive.google.com/drive/folders/1gd0EpqDv6b5N_zlLkw08do1kaFoK6ipN?usp=sharing)

## Acknowledgment
This work has been partially supported by NSERC and the 
National Research Council of Canada's Ideation Fund, New Beginnings Initiative. 
