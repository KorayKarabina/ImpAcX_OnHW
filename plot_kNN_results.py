import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from utils import config


# Set size function: For making publication quality plots: from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim



def get_acc_scores(input_str, n_list, n_id):
    """Gets kNN accuracy scores for each level in n_list

    Parameters
    ----------
    input_str: string
            Input string for the list of scores over 5 fold cross validation
    n_list: list
            List of various levels of n_significant or n_components to examine

    Returns
    -------
    mean_dict: Dictionary of mean accuracy scores over 5 fold cross validation
    std_dict: Dictionary of standard deviations over 5 fold cross validation
    """
    mean_dict = {}
    std_dict = {}



    for n in n_list:
        scores_list_all = []
        
        for k_fold_number in config.OnHW_FOLD:
            filepath = os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, f"{input_str}_fold{k_fold_number}_{n_id}{n}.txt")

            with open(filepath, "rb") as fp:
                scores_list_all.append(pickle.load(fp))


        mean_list = []
        std_list = []

        for i in range(1, 50):
            mean_list.append((scores_list_all[0][i] + scores_list_all[1][i] + scores_list_all[2][i] + scores_list_all[3][i] + scores_list_all[4][i])/5)
            std_list.append(np.std([scores_list_all[0][i], scores_list_all[1][i], scores_list_all[2][i], scores_list_all[3][i], scores_list_all[4][i]]))


        mean_dict[n] = mean_list
        std_dict[n] = std_list
        

    return mean_dict, std_dict




def plot_acc(ax, mean_dict, std_dict, n_list, k_range, color, lab):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    ax: plt.subplots()
            Object to plot on
    mean_dict: list
            Dictionary of mean accuracy scores over 5 fold cross validation
    std_dict: list
            Dictionary of standard deviations over 5 fold cross validation
    n_list: list
            List of various levels of n_significant or n_components to examine
    k_range: range()
            Range object representing the range of k values in the kNN
    color: list
            List of colors associated with the line color of each level in n_list
            
    Returns
    -------
    None
    """

    count = 0
    for n in n_list:
        mean_arr = np.asarray(mean_dict[n])
        std_arr = np.asarray(std_dict[n])

        ax.plot(k_range, mean_arr, color=color[count], label=lab + str(n)) # Producing mean line
        # plt.fill_between(k_range, mean_arr - std_arr, mean_arr + std_arr, color=color[count],alpha=0.4) # Filling cloud

        count += 1





# Creating plot with proper size
width = 500
fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1))



# Creating n_sig plot

# Creating plot with proper size
width = 500
fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1))


n_sig_list = config.NSIG_LIST # List of n_sig's to compute 
color = ['plum', 'gold', 'fuchsia', 'orchid', 'limegreen', 'red', 'navy','aqua', 'steelblue', 'slateblue'] # Corresponding color for each n_sig ----- Specify
k_range = range(1,50)

# Getting the accuracy scores
# input_str = "NCA_kNN"
input_str = "kNN"
mean_dict, std_dict = get_acc_scores(input_str, n_sig_list, n_id="nsig")
    
plot_acc(ax, mean_dict, std_dict, n_sig_list, k_range, color, lab='nsig=')
ax.set_xlabel('Value of k in kNN')
ax.set_ylabel('Average Testing Accuracy (5-fold Cross Validation)')
ax.set_title('kNN Performance')
ax.legend(loc = 'lower right')

# # Saving nsig plot
fig.savefig(os.path.join(config.BASE_OUTPUT, config.VISUALS, 'kNN_nsig_figure_colored'), format='pdf', bbox_inches='tight')









# Creating ncomponents plot
fig2, ax2 = plt.subplots(1, 1, figsize=set_size(width, fraction=1))

n_comp_list = config.NCOMP_LIST # List of n_comp's to compute
color = ['plum', 'gold', 'fuchsia', 'red', 'orchid', 'limegreen', 'navy','aqua', 'steelblue', 'slateblue', 'purple'] # Corresponding color for each n_sig ----- Specify

input_str = "NCA_kNN"
mean_dict, std_dict = get_acc_scores(input_str, n_comp_list, n_id="ncomp")

plot_acc(ax2, mean_dict, std_dict, n_comp_list, k_range, color, lab='ncomp=')
ax2.set_xlabel('Value of k in kNN')
ax2.set_ylabel('Average Testing Accuracy (5-fold Cross Validation)')
ax2.legend(loc = 'lower right')
ax2.set_title('NCA+kNN Performance')

# Saving ncomponents plot
fig2.savefig(os.path.join(config.BASE_OUTPUT, config.VISUALS, 'kNN_ncomp_figure_colored.pdf'), format='pdf', bbox_inches='tight')

