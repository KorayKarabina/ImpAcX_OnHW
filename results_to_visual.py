import os
import numpy as np
import pandas as pd
from utils import config, folders_and_files

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly.validators.scatter.marker import SymbolValidator

def results_to_mean_acc(results_file_name):
    df_results = pd.read_csv(results_file_name)
    groupby = df_results.groupby(['model', 'case', 'dependency'])
    results_dict = {'model':[], 'case': [], 'dependency':[], 'mean': [], 'std': []}
    for name, group in groupby:
        results_dict['model'].append(name[0])
        results_dict['case'].append(name[1])
        results_dict['dependency'].append(name[2])
        results_dict['mean'].append(group['accuracy'].mean())
        results_dict['std'].append(group['accuracy'].std())

    df = pd.DataFrame(results_dict)
    df.sort_values(by=['mean', 'case','dependency','model','std'], ascending=False, inplace=True)

    return df.reset_index(drop=True)

def results_to_fold_acc(results_file_name):
    df_results = pd.read_csv(results_file_name)
    groupby = df_results.groupby(['model', 'case', 'dependency', 'fold'])
    results_dict = {'model':[], 'case': [], 'dependency':[], 'fold':[], 'accuracy': []}
    for name, group in groupby:
        results_dict['model'].append(name[0])
        results_dict['case'].append(name[1])
        results_dict['dependency'].append(name[2])
        results_dict['fold'].append(name[3])
        results_dict['accuracy'].append(group['accuracy'].values[0])

    df = pd.DataFrame(results_dict)
    df.sort_values(by=['model','case','dependency','fold', 'accuracy'], inplace=True)
    df.reset_index(drop=True)

    return df.reset_index(drop=True)


def plot_all_results():
    ml_results_file_name =  os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, config.ML_RESULTS_CSV)
    df_ml_results_mean = results_to_mean_acc(ml_results_file_name)
    df_ml_results_fold = results_to_fold_acc(ml_results_file_name)
    df_ml_results_mean.to_csv(os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, config.ML_RESULTS_MEAN_CSV), index=False)
    df_ml_results_fold.to_csv(os.path.join(config.BASE_OUTPUT, config.ML_RESULTS, config.ML_RESULTS_FOLD_CSV), index=False)

    dl_results_file_name =  os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, config.DL_RESULTS_CSV)
    df_dl_results_mean = results_to_mean_acc(dl_results_file_name)
    df_dl_results_fold = results_to_fold_acc(dl_results_file_name)
    df_dl_results_mean.to_csv(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, config.DL_RESULTS_MEAN_CSV), index=False)
    df_dl_results_fold.to_csv(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS, config.DL_RESULTS_FOLD_CSV), index=False)

    dl_results_opt_file_name =  os.path.join(config.BASE_OUTPUT, config.DL_RESULTS_OPT, config.DL_RESULTS_OPT_CSV)
    df_dl_results_opt_mean = results_to_mean_acc(dl_results_opt_file_name)
    df_dl_results_opt_fold = results_to_fold_acc(dl_results_opt_file_name)
    df_dl_results_opt_mean.to_csv(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS_OPT, config.DL_RESULTS_MEAN_OPT_CSV), index=False)
    df_dl_results_opt_fold.to_csv(os.path.join(config.BASE_OUTPUT, config.DL_RESULTS_OPT, config.DL_RESULTS_FOLD_OPT_CSV), index=False)


    df_dl = df_dl_results_mean.copy()
    df_dl_opt = df_dl_results_opt_mean.copy()
    df_ml = df_ml_results_mean.copy()

    fig = make_subplots(rows=1, cols=1, print_grid=True)

    # models are sorted in a descending order w.r.t mean accuracy
    groupby_ml = df_ml.groupby("model")
    model_list_ml = list(groupby_ml['mean'].max().sort_values(ascending=False).index)

    groupby_dl = df_dl.groupby("model")
    model_list_dl = list(groupby_dl['mean'].max().sort_values(ascending=False).index)

    groupby_dl_opt = df_dl_opt.groupby("model")
    model_list_dl_opt = list(groupby_dl_opt['mean'].max().sort_values(ascending=False).index)

    sorted_dl_cases = (df_dl['case'] + '_' + df_dl['dependency']).unique()
    dl_cases_dict = dict(zip(sorted_dl_cases, np.arange(len(sorted_dl_cases))))

    sorted_dl_opt_cases = (df_dl_opt['case'] + '_' + df_dl_opt['dependency']).unique()
    shift_by = 0.2
    dl_opt_cases_dict = dict(zip(sorted_dl_opt_cases, [dl_cases_dict[k]+shift_by for k in sorted_dl_opt_cases]))
    # dl_opt_cases_dict = dict(zip(sorted_dl_opt_cases, np.arange(len(sorted_dl_opt_cases))))

    sorted_ml_cases = (df_ml['case'] + '_' + df_ml['dependency']).unique()
    ml_cases_dict = dict(zip(sorted_ml_cases, [dl_cases_dict[k]+2*shift_by for k in sorted_ml_cases]))


    raw_symbols = SymbolValidator().values

    dl_symbols = {}
    for ind, model in enumerate(model_list_dl):
       dl_symbols.update({model: raw_symbols[12*ind]})

    dl_opt_symbols = {}
    for ind, model in enumerate(model_list_dl_opt):
       dl_opt_symbols.update({model: raw_symbols[12*ind]})

    ml_symbols = {}
    for ind, model in enumerate(model_list_ml):
       ml_symbols.update({model: raw_symbols[12*ind]})



    for model in model_list_dl:
        group = groupby_dl.get_group(model)
        case_list = group['case'] + "_" + group['dependency']
        fig.add_trace(
                go.Scatter(
                    x=[dl_cases_dict[k] for k in case_list],
                    y=group['mean'],
                    name=model,
                    mode='markers',
                    marker_symbol=dl_symbols[model],
                    legendgroup='1',
                    legendgrouptitle_text="Deep Learning Models",
                    hovertemplate="Model=%s<br>Case=%%{x}<br>Accuracy=%%{y}<extra></extra>"% model),
                    secondary_y=False, row=1, col=1)

    for model in model_list_ml:
            group = groupby_ml.get_group(model)
            group['case'] + "_" + group['dependency']
            fig.add_trace(
                    go.Scatter(
                        x=[ml_cases_dict[k] for k in case_list],
                        y=group['mean'],
                        name=model,
                        mode='markers',
                        marker_symbol=ml_symbols[model],
                        legendgroup='2',
                        legendgrouptitle_text="Machine Learning Models",
                        hovertemplate="Model=%s<br>Case=%%{x}<br>Accuracy=%%{y}<extra></extra>"% model),
                        secondary_y=False, row=1, col=1)


    for model in model_list_dl_opt:
        group = groupby_dl_opt.get_group(model)
        case_list = group['case'] + "_" + group['dependency']
        fig.add_trace(
                go.Scatter(
                    x=[dl_opt_cases_dict[k] for k in case_list],
                    y=group['mean'],
                    name=model,
                    mode='markers',
                    marker_symbol=dl_opt_symbols[model],
                    legendgroup='3',
                    legendgrouptitle_text="Optimized Deep Learning Models",
                    hovertemplate="Model=%s<br>Case=%%{x}<br>Accuracy=%%{y}<extra></extra>"% model),
                    secondary_y=False, row=1, col=1)    


    fig.update_layout(
        height=900,
        width=1800,
        title_text="Mean Accuracy of Deep Learning and Machine Learning Models",
        yaxis1_title = 'Accuracy',
        legend_tracegroupgap = 90,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ),
        font=dict(
            family="Arial, sans-serif",
            size=15,
            color="black"
        )
    )

    x_tick_values = [dl_cases_dict[k]+shift_by//2 for k in sorted_dl_cases]
    tick_text = list(dl_cases_dict.keys())
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_tick_values,
            ticktext=tick_text,
        )
    )

    fig.update_annotations(font_size=15)
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Arial, sans-serif', color='black', size=15))
    fig.update_traces(marker_size=15)
    # fig.show()

    folders_and_files.make_folder_at(config.BASE_OUTPUT, config.VISUALS)
    fig.write_html(os.path.join(config.BASE_OUTPUT, config.VISUALS, 'ml_dl_all_results.html'))
    # fig.write_image(os.path.join(config.BASE_OUTPUT, config.VISUALS, 'ml_dl_all_results.png'))


if __name__ == "__main__":
    plot_all_results()