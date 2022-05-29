## Import libraries
import numpy as np
import pandas as pd
import logging
import h2o
from matplotlib import pyplot as plt

## Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Useful links
## H2O with monotonic constraings:
# https://medium.com/analytics-vidhya/application-of-monotonic-constraints-in-machine-learning-models-334564bea616
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/monotone_constraints.html


def ds_common_h2o_process_model_sub_unseen_data_performance(model_h2o, lst_performance_metrics, df_h2o, str_label):

    """
    This function process the H2O model_performance output of a single H2O model.
    It is strictly dependent on the way H2O returns the output (for instance some metrics are upper case, other are lower case) and, as of now, and only certain KPIs are available.
    In general this function returns a pandas frame with a "model_id" column and a set of model_performance KPIs extracted from the H2O model_performance function.
    
    Main steps:
    - Get all model_performances metrics from the df_h2o frame using model_h2o.
    - Filter metrics using lst_performance_metrics 
    - Rename columns using a suffix (e.g. '_train' or '_test')
    
    
    Parameters
    ----------
    model_h2o : H2O model
    
    lst_performance_metrics : list of strings
                              Each element is a performance metric of interest, for instance ['mae', 'mse'].
                              Remark: h2o model_performance may not have all performance metrics of interest.
                              
    df_h2o : H2O frame
    
    str_label : string
                Suffix used to rename each metric in lst_performance_metrics
                
    
    Returns
    ----------
    pandas frame
    
    """

    # Initialize output dictionary (that will be converted as pandas frame at the end)
    dict_out    = {'model_id': model_h2o.model_id}
    
    # Check if H2O frame is empty
    if(df_h2o.dim[0] < 1):
        logging.info(f'H2O Frame has no rows for label {str_label}, hence skipping everything')
        return pd.DataFrame(dict_out, index = [0])
        
    # Get H2O model performances from the df_h2o frame using model_h2o
    dict_data_performance = model_h2o.model_performance(df_h2o)._metric_json
    
    # Create output dictionary 
    try:
        # Initialize the column rename dictionary (e.g. adding '_train' or '_test' to the model_performance metrics such as MSE)
        dict_rename = {}
        for metric in lst_performance_metrics:    
            try:
                dict_out[metric] = dict_data_performance[metric]
            except Exception as e:
                try:
                    dict_out[metric] = dict_data_performance[metric.upper()]       
                except:
                    logging.info(f'General error for label {str_label}, model_id {model_h2o.model_id}, metric {metric} in the function ds_common_h2o_process_model_sub_unseen_data_performance')
            dict_rename[metric] = metric + f"_{str_label}"
          
        df_out = pd.DataFrame(dict_out, index = [0])       
        df_out.rename(columns = dict_rename, inplace = True)
        return df_out
    except Exception as e:
        logging.info(f'General error for label {str_label} and model_id {model_h2o.model_id} in the function ds_common_h2o_process_model_sub_unseen_data_performance')
        return pd.DataFrame(dict_out, index = [0])


def ds_common_h2o_process_model_sub_cross_validation_metrics_summary(model_h2o, lst_performance_metrics_lb):
    
    """
    This function process the H2O cross_validation_metrics_summary output of a single H2O model.
    It is strictly dependent on the way H2O returns the output (for instance the metric column is called '') as of now, and only certain KPIs are available.
    In general this function returns a pandas frame with a "model_id" column and a set of CV summary KPIs extracted from the H2O cross_validation_metrics_summary object.
    
    Parameters
    ----------
    model_h2o : H2O model
    
    lst_performance_metrics : list of strings
                              Each element is a performance metric of interest, for instance ['mae', 'mse'].
                              Remark: h2o cross_validation_metrics_summary may not have all metrics of interest.
                
    
    Returns
    ----------
    pandas frame
    
    """

    ## Extract H2O CV metrics summary
    df_cv_summary = model_h2o.cross_validation_metrics_summary().as_data_frame().reset_index(drop = True)
    
    ## Fix the metric column (H2O sets the name to '')
    df_cv_summary.rename(columns = {'': 'metric', 'mean': 'mean_cv', 'sd': 'sd_cv'}, inplace = True)
    
    ## Filter the metrics of interest (remark: this object may not have all metrics)
    df_cv_summary = df_cv_summary[df_cv_summary['metric'].isin(lst_performance_metrics_lb)].copy()
    
    ## Extra min/max cv value
    df_cv_summary['min_cv_value'] = df_cv_summary[[col for col in df_cv_summary.columns.tolist() if 'valid' in col]].min(axis = 1)
    df_cv_summary['max_cv_value'] = df_cv_summary[[col for col in df_cv_summary.columns.tolist() if 'valid' in col]].max(axis = 1)
    
    ## Add model_id
    df_cv_summary['model_id']     = model_h2o.model_id
    
    ## Pivot table for having everything on 1 row
    df_cv_summary = df_cv_summary.pivot(index   = 'model_id',
                                        columns = 'metric',
                                        values  = [col for col in df_cv_summary.columns.tolist() if 'metric' not in col]).reset_index(drop = True)
    
    ## Rename columns
    df_cv_summary.columns = df_cv_summary.columns.get_level_values(0) + '_' + df_cv_summary.columns.get_level_values(1)
    
    ## Since the .pivot adds useless columns such as model_id_mae and model_id_mse, fix it.
    df_cv_summary             = df_cv_summary[[col for col in df_cv_summary.columns.tolist() if 'model_id' not in col]].copy()
    df_cv_summary['model_id'] = model_h2o.model_id
    
    return df_cv_summary


def ds_common_h2o_process_model_sub_leaderboard(df_automl, str_model_id, lst_performance_metrics_lb, lst_other_metrics_lb):
    
    """
    This function process the H2O leaderboard output of an Automl H2O object.
    It is strictly dependent on the way H2O returns the output.
    In general this function returns a pandas frame with a "model_id" column and a set of leaderboard KPIs.   
    
    Parameters
    ----------
    df_automl : pandas frame
                Obtained by converting the H2O leaderboard to a pandas frame.
                   
    str_model_id : string
                   Name of the model_id to be processed.
    
    lst_performance_metrics_lb : list of strings
                                 Each element is a performance metric of interest, for instance ['mae', 'mse'].
                                 Remark: h2o leaderboard may not have all metrics of interest.

    lst_other_metrics_lb : list of strings
                           Each element is a non-metric of interest, for instance ['training_time_ms', 'predict_time_per_row_ms', 'algo', 'model_id'].
                           Remark: h2o leaderboard may not have all metrics of interest.
    
    Returns
    ----------
    pandas frame
    
    """
    
    # Filter rows
    df_automl_cv = df_automl[df_automl['model_id']==str_model_id].copy()
    
    # Filter columns
    df_automl_cv               = df_automl_cv[lst_performance_metrics_lb + lst_other_metrics_lb].copy()
    
    # Rename list_performance_metrics_regression_lb
    dict_rename = {}
    for col in lst_performance_metrics_lb:
        dict_rename[col] = col + "_lb"
    df_automl_cv.rename(columns = dict_rename, inplace = True)
    
    return df_automl_cv


def ds_common_h2o_process_automl_main(automl_h2o, df_h2o_train, df_h2o_test, df_h2o_valid, str_tag_data_experiment, str_tag_model_experiment, str_ml_problem_type):
    
    """
    This function process an H2O Automl object and returns KPIs related to:
    - Train, test and cross-validation metrics.
    - Model hyperparameters.
    - Model training & scoring time.
    
    Furthermore, it also saves model artifacts locally.
    
    
    Parameters
    ----------
    automl_h2o : H2O Automl
                 Trained H2O Automl object
                   
    df_h2o_train : H2O Frame
    
    df_h2o_test : H2O Frame
    
    df_h2o_valid : H2O Frame
    
    str_tag_data_experiment : string
    
    str_tag_model_experiment : string
                              
    str_ml_problem_type : string
                          Currently not used, might be useful for the future. Potential values are:
                          - 'regression'
                          - 'binary_classification'
                          - 'multinomial_classification'

    
    Returns
    ----------
    df_output_info : pandas frame
                     Containing model information such as hyperparameters, performance metrics on train, test, validation and cross-validation sets.
    
    Open points TODO
    - Should we return also model predictions? -> No
    """
    
    ## Set global parameters
    # H2O Leaderboard frame column names to be extracted
    lst_other_metrics = ['training_time_ms', 'predict_time_per_row_ms', 'algo', 'model_id']
                           
    
    ## The following lists are taken from https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
    # They are the only hyperparameters searched by H2O Automl
    lst_hyperparams_gbm = [
                           'col_sample_rate', 'col_sample_rate_per_tree','max_depth', 'min_rows', 'min_split_improvement',
                           'ntrees', 'sample_rate'
                          ]
    
    lst_hyperparams_xgb = [
                           'booster', 'col_sample_rate', 'col_sample_rate_per_tree','max_depth', 'min_rows', 'ntrees',
                           'sample_rate', 'reg_alpha', 'reg_lambda'
                          ]    
    
    lst_hyperparams_deeplearning = [
                                    'epochs', 'epsilon', 'hidden', 'hidden_dropout_ratios', 'input_dropout_ratio', 'rho'
                                   ] 
    
    
    ## Get Leaderboard
    df_automl_cv = h2o.automl.get_leaderboard(automl_h2o, extra_columns = "ALL").as_data_frame()
    
    
    ## Get lists of performance metrics for the H2O cv_summary object, H2O leaderboard object and H2O model_performance object
    lst_performance_metrics, lst_performance_metrics_cv_summary, lst_performance_metrics_model_performance = get_h2o_performance_metrics_lists(str_ml_problem_type)
    
    
    ## Extract Models information
    list_df = []
    for str_model_id in df_automl_cv['model_id'].tolist():
        
        # Get model
        model_h2o = h2o.get_model(str_model_id)
        
        # Get LB info
        df_lb = ds_common_h2o_process_model_sub_leaderboard(df_automl_cv, str_model_id, lst_performance_metrics, lst_other_metrics)
        
        # Get CV info
        df_cv = ds_common_h2o_process_model_sub_cross_validation_metrics_summary(model_h2o, lst_performance_metrics_cv_summary)
        
        # Get Train info
        df_perf_train = ds_common_h2o_process_model_sub_unseen_data_performance(model_h2o, lst_performance_metrics_model_performance, df_h2o_train, 'train')

        # Get Test info
        df_perf_test = ds_common_h2o_process_model_sub_unseen_data_performance(model_h2o, lst_performance_metrics_model_performance, df_h2o_test, 'test')
        
        # Get Validation info
        df_perf_valid = ds_common_h2o_process_model_sub_unseen_data_performance(model_h2o, lst_performance_metrics_model_performance, df_h2o_valid, 'valid')
        
        # Get Model info
        str_algo = df_automl_cv[df_automl_cv['model_id']==str_model_id].copy()
        str_algo = str_algo.reset_index(drop = True)
        str_algo = str_algo['algo'][0]
        str_algo = str_algo.lower()
        if(str_algo=='gbm'):
            df_hyperparams = ds_common_h2o_process_model_sub_get_info_gbm_xgb(model_h2o, lst_hyperparams_gbm)
        elif(str_algo=='xgboost'):   
            df_hyperparams = ds_common_h2o_process_model_sub_get_info_gbm_xgb(model_h2o, lst_hyperparams_xgb)
        elif(str_algo=='glm'):
            df_hyperparams = ds_common_h2o_process_model_sub_get_info_glm(model_h2o)
        elif(str_algo=='deeplearning'):
            df_hyperparams = ds_common_h2o_process_model_sub_get_info_deep_learning(model_h2o, lst_hyperparams_deeplearning)
        elif(str_algo=='stackedensemble'):
            df_hyperparams = ds_common_h2o_process_model_sub_get_info_stacked_ensemble(model_h2o)
             
        # Join all Info and update the dataframe list
        df_out = pd.merge(left = df_lb,  right = df_cv,           how = "left", on = 'model_id')
        df_out = pd.merge(left = df_out, right = df_perf_train,   how = "left", on = 'model_id')
        df_out = pd.merge(left = df_out, right = df_perf_test,    how = "left", on = 'model_id')
        df_out = pd.merge(left = df_out, right = df_perf_valid,   how = "left", on = 'model_id')
        df_out = pd.merge(left = df_out, right = df_hyperparams,  how = "left", on = 'model_id')
        
        # Update lists of dataframes
        list_df.append(df_out)
    
    # Prepare Automl output metrics dataframe
    df_output_info                         = pd.concat(list_df)
    df_output_info['tag_data_experiment']  = str_tag_data_experiment
    df_output_info['tag_model_experiment'] = str_tag_model_experiment
    
    # Sort columns
    df_output_info = df_output_info[['tag_data_experiment', 'tag_model_experiment', 'model_id', 'algo'] + [col for col in df_output_info.columns.tolist() if col != 'model_id' and col != 'algo' and col != 'tag_data_experiment' and col != 'tag_model_experiment']].copy()
    
    # Create a df index
    df_output_info        = df_output_info.reset_index(drop = True)
    
    # Lower column names
    df_output_info.columns = [x.lower() for x in df_output_info.columns.tolist()]
    
    return df_output_info
  
def process_feature_importance(str_model_id, df_feature_importance, int_number_features):
    
    """
    This function process a pandas frame containing H2O Feature importance.
    It returns a pandas frame with the following columns:
    - model_id
    - number_variables
    - number_features
    - str_feature_importance
    
    
    Parameters
    ----------
    str_model_id : string
                   
    df_feature_importance : pandas Frame
                            Extracted from H2O.
    
    int_number_features : integer
                          Number of features to be merged in the final string.

    
    Returns
    ----------
    df_feature_importance : pandas frame
                            Containing feature importance information aggregated by the model_id.
    """
    
    
    ## Add model_id
    df_feature_importance['model_id'] = str_model_id
    
    ## Remove features with 0 importance
    df_feature_importance            = df_feature_importance[df_feature_importance['percentage'] > 0].copy()
    
    ## Create feature column
    df_feature_importance['feature'] = df_feature_importance['variable'].map(lambda x: x.split(".")[0])
    
    ## Filter rows
    df_feature_importance = df_feature_importance[df_feature_importance.index <=int_number_features].copy()
    number_variables      = len(set(df_feature_importance['variable']))
    number_features       = len(set(df_feature_importance['feature']))
    
    ## Create concat column
    df_feature_importance['str_feature_importance'] = df_feature_importance['variable'] + '_' + df_feature_importance['percentage'].map(lambda x: str(round(x, 4)))
    
    ## Filter columns
    df_feature_importance = df_feature_importance[['model_id', 'str_feature_importance']].copy()
    
    ## Aggregate string
    df_feature_importance = df_feature_importance.groupby(by = 'model_id').agg({'str_feature_importance': ', '.join}).reset_index()
    
    # Add features
    df_feature_importance['number_variables'] = number_variables
    df_feature_importance['number_features']  = number_features
    
    return df_feature_importance

def ds_common_h2o_process_model_sub_get_info_stacked_ensemble(model_h2o):
    
    """
    This function process the output of a single H2O stacked ensemble model.
    
    Parameters
    ----------
    model_h2o : H2O model

    
    Returns
    ----------
    df_out_hyper : pandas frame
    """
    
    df_out_hyper = pd.DataFrame({'model_id': model_h2o.model_id}, index = [0])
    
    return df_out_hyper

def ds_common_h2o_process_model_sub_get_info_deep_learning(model_h2o, list_hyperparams):
    
    """
    This function process the output of a single H2O deep learning model, with a focus on:
    - feature importance
    - hyperparameters
    
    Parameters
    ----------
    model_h2o : H2O model
    
    list_hyperparams : list of strings
                       Each element is an hyperparameter tuned by H2O. More details can be found
    
    Returns
    ----------
    df_out_hyper : pandas frame
    """
    
    ## Extract model output
    dict_output = model_h2o._model_json["output"]
    
    ## Extract variable importance (only top 10 features)
    top_n_features        = 10
    df_feature_importance = process_feature_importance(model_h2o.model_id, dict_output['variable_importances'].as_data_frame(), top_n_features)
    
    ## Initialize output dictionary
    dict_out_hyper = {'model_id': model_h2o.model_id}
    
    ## Extract hyperparameters
    for hyperparameter in list_hyperparams:
        if(type(model_h2o.params[hyperparameter]['actual'])==list):
            dict_out_hyper[hyperparameter] = ', '.join([str(x) for x in model_h2o.params[hyperparameter]['actual']])
        else:
            dict_out_hyper[hyperparameter] = model_h2o.params[hyperparameter]['actual']
    
    df_out_hyper = pd.DataFrame(dict_out_hyper, index = [0])
    df_out_hyper = pd.merge(left = df_out_hyper, right = df_feature_importance, how = "left", on = 'model_id')
    
    return df_out_hyper


def ds_common_h2o_process_model_sub_get_info_gbm_xgb(model_h2o, list_hyperparams):
    
    """
    This function process the output of a single H2O XGB or GBM models, with a focus on:
    - feature importance
    - hyperparameters
    
    Parameters
    ----------
    model_h2o : H2O model
    
    list_hyperparams : list of strings
                       Each element is an hyperparameter tuned by H2O. More details can be found
    
    Returns
    ----------
    df_out_hyper : pandas frame
    """
    
    ## Extract model output
    dict_output = model_h2o._model_json["output"]
    
    ## Extract variable importance (only top 10 features)
    top_n_features        = 10
    df_feature_importance = process_feature_importance(model_h2o.model_id, dict_output['variable_importances'].as_data_frame(), top_n_features)
    
    ## Extract hyperparameters
    dict_out_hyper = {'model_id': model_h2o.model_id}
    for hyperparameter in list_hyperparams:
        if(type(model_h2o.params[hyperparameter]['actual'])==list):
            dict_out_hyper[hyperparameter] = ', '.join([str(x) for x in model_h2o.params[hyperparameter]['actual']])
        else:
            dict_out_hyper[hyperparameter] = model_h2o.params[hyperparameter]['actual']
            
    df_out_hyper = pd.DataFrame(dict_out_hyper, index = [0])
    df_out_hyper = pd.merge(left = df_out_hyper, right = df_feature_importance, how = "left", on = 'model_id')
    
    return df_out_hyper


def ds_common_h2o_process_model_sub_get_info_glm(model_h2o):
    
    """
    This function process the output of a single H2O GLM model, with a focus on:
    - feature importance
    - hyperparameters
    
    Parameters
    ----------
    model_h2o : H2O model
    
    Returns
    ----------
    df_out_hyper : pandas frame
    """
    
    ## Extract model output
    dict_output = model_h2o._model_json["output"]
    
    ## Extract variable importance (only top 10 features)
    top_n_features        = 10
    df_feature_importance = process_feature_importance(model_h2o.model_id, dict_output['variable_importances'].as_data_frame(), top_n_features)
    
    ## Extract hyperparameters (for GLM it is simpler to extract them in this way instead of using list_hyperparams)
    dict_out_hyper = {
                      'model_id': model_h2o.model_id, 
                      'alpha':    dict_output['alpha_best'],
                      'lambda':   dict_output['lambda_best']
                     }
    
    df_out_hyper = pd.DataFrame(dict_out_hyper, index = [0])
    df_out_hyper = pd.merge(left = df_out_hyper, right = df_feature_importance, how = "left", on = 'model_id')
    
    return df_out_hyper

def ds_common_h2o_plot_automl_hyperparams(df_output_info, lst_hyperparams, lst_str_type,
                                          str_selected_metric, str_selected_algo, str_path_output_figure):
    
    """
    This function plots and locally saves a set of diagnostic plot of an Automl Output object.
    Main steps:
    - Filter pandas Automl output by the algorithm of choice (-> Xgboost, GBM, Deep learning).
    - Plot the correlation of each pair of (hyperparam, str_selected_metric_<lst_str_type>). 
    - Plot CV vs Test str_selected_metric value.
    - Locally save all plots in a single picture.
    - Show the plot.
    
    
    Parameters
    ----------
    df_output_info : pandas frame
                     Output of ds_common_h2o_process_automl_main
    
    lst_hyperparams : list of strings
                      Each element is an hyperparameter
    
    lst_str_type : list of strings
                   Each element is a level of the fold, e.g. 'lb' or 'test' or 'train'. 
    
    str_selected_metric : string
                          Name of the metric used for plots. For instance 'mae' or 'mse'.  
    
    str_selected_algo : string
                        Name of the (h2o) algorithm used for the plot. Either 'GBM', 'Xgboost' or 'DeepLearning'.
    
    str_path_output_figure : string
                             Local path + name of the file to be saved for the plot.
    
    Returns
    ----------
    -
    """
    
    # Filter output dataframe
    df_output_info_filtered = df_output_info[df_output_info['algo']==str_selected_algo].copy()
    
    # Plot correlation between hyperparameters values and selected metric
    fig, ax = plt.subplots(nrows=len(lst_hyperparams) + 1, ncols=len(lst_str_type), figsize=(15,15))
    fig.suptitle('Correlation between hyperparameter value and RMSE CV / test')
    
    cont_row = 0
    for param in lst_hyperparams:
        cont_col = 0
        for str_fold in lst_str_type:
            ax[cont_row, cont_col].scatter(df_output_info_filtered[param], df_output_info_filtered[f'{str_selected_metric}_{str_fold}'])
            ax[cont_row, cont_col].set_title(f'Hyperparam: {param}; Metric: {str_selected_metric}; Fold: {str_fold}')
            cont_col = cont_col + 1
        cont_row = cont_row + 1
                
    # Plot correlation between lst_str_type and the metric str_selected_metric
    ax[cont_row, 0].scatter(df_output_info_filtered[f'{str_selected_metric}_lb'], df_output_info_filtered[f'{str_selected_metric}_test'])
    ax[cont_row, 0].set_title(f'CV vs Test {str_selected_metric} value')
    
    # Save figure
    plt.savefig(f'{str_path_output_figure}', bbox_inches='tight')
    
    # Show figure
    fig.tight_layout()
    plt.show()
    
    return


def ds_common_h2o_plot_model_explaination_sub(model_h2o, df_h2o, str_path_root_output, lst_column_plot, str_fold):

    """
    This function create and saves various plots for explainability analysis leveraging an H2O Model Object and an H2O dataframe.
    
    
    Parameters
    ----------
    model_h2o : H2O Model
    
    df_h2o : H2O Frame
    
    lst_str_type : list of strings
                   Each element is a level of the fold, e.g. 'lb' or 'test' or 'train'. 
        
    str_path_root_output : string
                           Local root path where plots must be saved, for instance 'plot/'. 
    
    lst_column_plot : list of strings  
                      Each element is a column name that will be used for partial dependence & ice plots.
    
    str_fold : string
               Name of the fold of df_h2o. Either 'train', 'test' or 'valid'. 
    
    
    Returns
    ----------
    -
    """
    
    # If H2O Frame has at least 1 record perform the analysis, else skip all of them
    if(df_h2o.dim[0] > 1):
        
        # Shap summmary
        try:
            model_h2o.shap_summary_plot(df_h2o, save_plot_path = f'{str_path_root_output}plot_shap_summary_{str_fold}_{model_h2o.model_id}.pdf')
        except Exception as e:
            logger.info(f'Fold {str_fold} and model {model_h2o.model_id} had errors during shap summary plot, skipping it.')
        
        # Residuals
        try:
            model_h2o.residual_analysis_plot(df_h2o, save_plot_path = f'{str_path_root_output}plot_residuals_{str_fold}_{model_h2o.model_id}.pdf')
        except Exception as e:
            logger.info(f'Fold {str_fold} and model {model_h2o.model_id} had errors during residuals plot, skipping it.')
        
        # Partial dependence and ICE
        for str_col in lst_column_plot:
            try:
                model_h2o.pd_plot(df_h2o, column = str_col, save_plot_path = f'{str_path_root_output}plot_pd_{str_fold}_{str_col}_{model_h2o.model_id}.pdf')
                model_h2o.ice_plot(df_h2o, column = str_col, save_plot_path = f'{str_path_root_output}plot_ice_{str_fold}_{str_col}_{model_h2o.model_id}.pdf')
            except Exception as e:
                logger.info(f'Fold {str_fold} and model {model_h2o.model_id} and column {str_col} had errors during pd or ice plots, skipping them.')
    else:
        logger.info(f'Fold {str_fold} has no rows - skipping all plots')
        
    return


def ds_common_h2o_plot_model_explaination_main(model_h2o, 
                                               df_h2o_train, df_h2o_test, df_h2o_valid,
                                               str_path_root_output, lst_column_plot):
    
    """
    This function calls various sub-functions for plotting explainability analysis leveraging an H2O Model Object and a set of H2O dataframes.

    Parameters
    ----------
    model_h2o : H2O Model
    
    df_h2o_train : H2O Frame 
    
    df_h2o_test : H2O Frame
    
    df_h2o_valid : H2O Frame
        
    str_path_root_output : string
                           Local root path where plots must be saved, for instance 'plot/'. 
    
    lst_column_plot : list of strings  
                      Each element is a column name that will be used for partial dependence & ice plots.
    
    
    Returns
    ----------
    -
    """
    
    # Train / Test / Validation diagnostic plots
    ds_common_h2o_plot_model_explaination_sub(model_h2o, df_h2o_train, str_path_root_output, lst_column_plot, 'train')
    ds_common_h2o_plot_model_explaination_sub(model_h2o, df_h2o_test, str_path_root_output, lst_column_plot, 'test')
    ds_common_h2o_plot_model_explaination_sub(model_h2o, df_h2o_valid, str_path_root_output, lst_column_plot, 'valid')
        
    # Shap row
    #model_h2o.shap_explain_row_plot(df_h2o_test, row_index = 0, save_plot_path = 'plots/plot_shap_row.pdf')
    
    # Learning curve
    try:
        model_h2o.learning_curve_plot(save_plot_path = f'{str_path_root_output}plot_learning_curve_{model_h2o.model_id}.pdf')
    except Exception as e:
        logger.info(f'Model {model_h2o.model_id} had errors during learning curve plots, skipping it')
    
    # Variable importance
    try:
        model_h2o.varimp_plot(save_plot_path = f'{str_path_root_output}plot_variable_importance_{model_h2o.model_id}.pdf')
    except Exception as e:
        logger.info(f'Model {model_h2o.model_id} had errors during variable importance plots, skipping it')
        
    return
        
def ds_common_h2o_plot_automl_explaination_sub(aml_h2o, df_h2o, str_path_root_output, lst_column_plot, str_fold,
                                               str_tag_data_experiment, str_tag_model_experiment):
    
    """
    This function create and saves various plots for explainability analysis leveraging an H2O Automl Object and an H2O dataframe.

    Parameters
    ----------
    model_h2o : H2O Model
    
    df_h2o : H2O Frame 
        
    str_path_root_output : string
                           Local root path where plots must be saved, for instance 'plot/'. 
    
    lst_column_plot : list of strings  
                      Each element is a column name that will be used for partial dependence & ice plots.
                      
    str_fold : string
               Name of the fold of df_h2o. Either 'train', 'test' or 'valid'.   
               
    str_tag_data_experiment : string
    
    str_tag_model_experiment : string
    
    
    Returns
    ----------
    -
    """
    
    # If H2O Frame has at least 1 record perform the analysis, else skip all of them
    if(df_h2o.dim[0] > 1):
        
        # Model correlation Heatmap
        try:
            h2o.model_correlation_heatmap(aml_h2o, df_h2o, save_plot_path = f'{str_path_root_output}/plot_automl_model_correlation_heatmap_{str_fold}_{str_tag_data_experiment}_{str_tag_model_experiment}.pdf')
        except Exception as e:
            logger.info(f'Fold {str_fold} error on model correlation heatmap, skipping it')
            
        # Partial dependence plots
        for str_col in lst_column_plot:
            try:
                h2o.pd_multi_plot(aml_h2o, df_h2o, column = str_col, save_plot_path = f'{str_path_root_output}/plot_automl_pdplot_{str_col}_{str_fold}_{str_tag_data_experiment}_{str_tag_model_experiment}.pdf')
            except Exception as e:
                logger.info(f'Fold {str_fold} and column {str_col} error on partial dependence plot, skipping it')   
    else:
        logger.info(f'Fold {str_fold} has no rows - skipping all plots')
    
    return
    

def ds_common_h2o_plot_automl_explaination_main(aml_h2o, 
                                                df_h2o_train, df_h2o_test, df_h2o_valid,
                                                str_path_root_output, lst_column_plot,
                                                int_number_features,
                                                str_tag_data_experiment, str_tag_model_experiment):
    """
    This function calls various sub-functions for plotting explainability analysis leveraging an H2O Automl Object and a set of H2O dataframes.

    Parameters
    ----------
    model_h2o : H2O Model
    
    df_h2o_train : H2O Frame 
    
    df_h2o_test : H2O Frame
    
    df_h2o_valid : H2O Frame
        
    str_path_root_output : string
                           Local root path where plots must be saved, for instance 'plot/'. 
    
    lst_column_plot : list of strings  
                      Each element is a column name that will be used for partial dependence & ice plots.     
    
    str_tag_data_experiment : string
    
    str_tag_model_experiment : string
    
    
    Returns
    ----------
    -
    """
    
    
    # Varimp Heatmap
    h2o.varimp_heatmap(aml_h2o, save_plot_path = f'{str_path_root_output}plot_automl_varimp_heatmap_{str_tag_data_experiment}_{str_tag_model_experiment}.pdf', num_of_features = int_number_features)

    
    # Model correlation heatmap & Partial dependence multi plots
    ds_common_h2o_plot_automl_explaination_sub(aml_h2o, df_h2o_train, str_path_root_output, lst_column_plot, 'train', str_tag_data_experiment, str_tag_model_experiment)
    ds_common_h2o_plot_automl_explaination_sub(aml_h2o, df_h2o_test, str_path_root_output, lst_column_plot, 'test', str_tag_data_experiment, str_tag_model_experiment)
    ds_common_h2o_plot_automl_explaination_sub(aml_h2o, df_h2o_valid, str_path_root_output, lst_column_plot, 'valid', str_tag_data_experiment, str_tag_model_experiment)    
    
    return


def get_h2o_performance_metrics_lists(str_ml_problem_type):
    
    """
    This function return three lists containing performance metrics to be extracted from various H2O Objects:
    - H2O Leaderboard
    - H2O Cross-validation summary
    - H2O Model performance
    
    After having fit an H2O Automl object, performance metrics for each object can be inspected by:
    
    # H2O Leaderboard metrics
    aml.leaderboard
    
    # H2O cross-validation summary metrics  
    h2o.get_model(aml.leaderboard.as_data_frame()['model_id'][0]).model_performance(df_h2o_test)._metric_json.keys()
    
    # H2O Model performance metrics
    h2o.get_model(aml.leaderboard.as_data_frame()['model_id'][0]).cross_validation_metrics_summary().as_data_frame().reset_index(drop = True)
    
    Custom metrics are not supported yet; more info can be found here:
    https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/custom_metric_func.html
    
    """
    
    # Regression problem performance metrics
    if(str_ml_problem_type == 'regression'):
        lst_performance_metrics_learboard         = ['mae', 'mse']
        lst_performance_metrics_cv_summary        = ['mae', 'mse']
        lst_performance_metrics_model_performance = ['mae', 'MSE']
        
    # Binary classification performance metrics
    elif(str_ml_problem_type == 'binary_classification'):
        lst_performance_metrics_learboard         = ['auc', 'logloss', 'aucpr']
        lst_performance_metrics_cv_summary        = ['AUC', 'pr_auc', 'Gini', 'f1', 'lift_top_group', 'logloss']
        lst_performance_metrics_model_performance = ['auc', 'pr_auc', 'logloss']
    
    # Multiclassification performance metrics
    elif(str_ml_problem_type == 'multinomial_classification'):
        lst_performance_metrics_learboard         = ['logloss']
        lst_performance_metrics_cv_summary        = ['logloss']
        lst_performance_metrics_model_performance = ['logloss']
    
    return lst_performance_metrics_learboard, lst_performance_metrics_cv_summary, lst_performance_metrics_model_performance
