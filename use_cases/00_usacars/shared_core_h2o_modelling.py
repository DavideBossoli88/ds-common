## Import libraries
import numpy as np
import pandas as pd
import logging
import h2o


## Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def ds_common_h2o_process_model_sub_unseen_data_performance(model_h2o, str_model_id, dict_h2o_process_automl, df_h2o, str_label):

    """
    This function process the H2O model_performance output of a single H2O model.
    It is strictly dependent on the way H2O returns the output (for instance some metrics are upper case, other are lower case) and, as of now, and only certain KPIs are available.
    In general this function returns a pandas frame with a "model_id" column and a set of model_performance KPIs.
    """
        
    # Extract model performances
    dict_data_performance = model_h2o.model_performance(df_h2o)._metric_json
    
    # Extra info from the json metric. At the moment some metrics are upper, others are lower.
    dict_out    = {'model_id': str_model_id}
    dict_rename = {}
    for metric in dict_h2o_process_automl['list_performance_metrics_regression_lb']:    
        try:
            dict_out[metric]    = dict_data_performance[metric]
        except:
            dict_out[metric]   = dict_data_performance[metric.upper()]  
        dict_rename[metric] = metric + f"_{str_label}"
      
    df_out = pd.DataFrame(dict_out, index = [0])       
    df_out.rename(columns = dict_rename, inplace = True)
    
    return df_out


def ds_common_h2o_process_model_sub_cross_validation_metrics_summary(model_h2o, str_model_id, dict_h2o_process_automl):
    
    """
    This function process the H2O cross_validation_metrics_summary output of a single H2O model.
    It is strictly dependent on the way H2O returns the output (for instance the metric column is called '') as of now, and only certain KPIs are available.
    In general this function returns a pandas frame with a "model_id" column and a set of CV summary KPIs.
    """

    ## Extract H2O CV metrics summary
    df_cv_summary = model_h2o.cross_validation_metrics_summary().as_data_frame().reset_index(drop = True)
    
    ## Fix the metric column (H2O sets the name to '')
    df_cv_summary.rename(columns = {'': 'metric', 'mean': 'mean_cv', 'sd': 'sd_cv'}, inplace = True)
    
    ## Filter the metrics of interest (remark: this object may not have all metrics)
    df_cv_summary = df_cv_summary[df_cv_summary['metric'].isin(dict_h2o_process_automl['list_performance_metrics_regression_lb'])].copy()
    
    ## Extra min/max cv value
    df_cv_summary['min_cv_value'] = df_cv_summary[[col for col in df_cv_summary.columns.tolist() if 'valid' in col]].min(axis = 1)
    df_cv_summary['max_cv_value'] = df_cv_summary[[col for col in df_cv_summary.columns.tolist() if 'valid' in col]].max(axis = 1)
    
    ## Add model_id
    df_cv_summary['model_id']     = str_model_id
    
    ## Pivot table for having everything on 1 row
    df_cv_summary = df_cv_summary.pivot(index   = 'model_id',
                                        columns = 'metric',
                                        values  = [col for col in df_cv_summary.columns.tolist() if 'metric' not in col]).reset_index(drop = True)
    
    ## Rename columns
    df_cv_summary.columns = df_cv_summary.columns.get_level_values(0) + '_' + df_cv_summary.columns.get_level_values(1)
    
    ## Since the .pivot adds useless columns such as model_id_mae and model_id_mse, fix it.
    df_cv_summary             = df_cv_summary[[col for col in df_cv_summary.columns.tolist() if 'model_id' not in col]].copy()
    df_cv_summary['model_id'] = str_model_id
    
    return df_cv_summary


def ds_common_h2o_process_model_sub_leaderboard(df_automl_cv, str_model_id, dict_h2o_process_automl):
    
    """
    This function process the H2O leaderboard output of an Automl H2O object.
    It is strictly dependent on the way H2O returns the output.
    In general this function returns a pandas frame with a "model_id" column and a set of leaderboard KPIs.   
    """
    
    # Filter rows
    df_automl_cv = df_automl_cv[df_automl_cv['model_id']==str_model_id].copy()
    
    # Filter columns
    df_automl_cv = df_automl_cv[dict_h2o_process_automl['list_performance_metrics_regression_lb'] + dict_h2o_process_automl['list_other_metrics_lb']].copy()
    
    # Rename list_performance_metrics_regression_lb
    dict_rename = {}
    for col in dict_h2o_process_automl['list_performance_metrics_regression_lb']:
        dict_rename[col] = col + "_lb"
    df_automl_cv.rename(columns = dict_rename, inplace = True)
    
    return df_automl_cv


def ds_common_h2o_process_automl_main(aml, df_h2o_train, df_h2o_test, dict_h2o_process_automl):
    
    """
    This function process an H2O Automl object and returns KPIs related to:
    - train, test and cross-validation metrics
    - model hyperparameters
    - model training & scoring time
    
    Furthermore, it also saves model artifacts locally.
    """
    
    
    ## Get Leaderboard
    df_automl_cv = h2o.automl.get_leaderboard(aml, extra_columns = "ALL").as_data_frame()
    
    
    ## Extract Models information
    list_df                    = []
    list_df_feature_importance = []
    for str_model_id in df_automl_cv['model_id'].tolist():
        
        # Get model
        model_h2o = h2o.get_model(str_model_id)
        
        # Get LB info
        df_lb = ds_common_h2o_process_model_sub_leaderboard(df_automl_cv, str_model_id, dict_h2o_process_automl)
        
        # Get CV info
        df_cv = ds_common_h2o_process_model_sub_cross_validation_metrics_summary(model_h2o, str_model_id, dict_h2o_process_automl)
        
        # Get Train info
        df_perf_train = ds_common_h2o_process_model_sub_unseen_data_performance(model_h2o, str_model_id, dict_h2o_process_automl, df_h2o_train, 'train')

        # Get Test info
        df_perf_test = ds_common_h2o_process_model_sub_unseen_data_performance(model_h2o, str_model_id, dict_h2o_process_automl, df_h2o_test, 'test')
        
        # Get Model info
        str_algo = df_automl_cv[df_automl_cv['model_id']==str_model_id].copy()
        str_algo = str_algo.reset_index(drop = True)
        str_algo = str_algo['algo'][0]
        str_algo = str_algo.lower()
        if(str_algo=='gbm'):
            df_feature_importance, df_hyperparams = ds_common_h2o_process_model_sub_get_info_gbm_xgb(str_model_id, model_h2o, dict_h2o_process_automl['list_hyperparams_gbm'])
        elif(str_algo=='xgboost'):   
            df_feature_importance, df_hyperparams = ds_common_h2o_process_model_sub_get_info_gbm_xgb(str_model_id, model_h2o, dict_h2o_process_automl['list_hyperparams_xgb'])
        elif(str_algo=='glm'):
            df_feature_importance, df_hyperparams = ds_common_h2o_process_model_sub_get_info_glm(str_model_id, model_h2o, dict_h2o_process_automl)
        elif(str_algo=='deeplearning'):
            df_feature_importance, df_hyperparams = ds_common_h2o_process_model_sub_get_info_deep_learning(str_model_id, model_h2o, dict_h2o_process_automl['list_hyperparams_deeplearning'])
        elif(str_algo=='stackedensemble'):
            df_feature_importance, df_hyperparams = ds_common_h2o_process_model_sub_get_info_stacked_ensemble(str_model_id)
             
        # Join all Info and update the dataframe list
        df_out = pd.merge(left = df_lb,  right = df_cv,           how = "left", on = 'model_id')
        df_out = pd.merge(left = df_out, right = df_perf_train,   how = "left", on = 'model_id')
        df_out = pd.merge(left = df_out, right = df_perf_test,    how = "left", on = 'model_id')
        df_out = pd.merge(left = df_out, right = df_hyperparams,  how = "left", on = 'model_id')
        list_df.append(df_out)
        list_df_feature_importance.append(df_feature_importance)
    
    df_output_info                         = pd.concat(list_df)
    df_output_info['tag_data_experiment']  = dict_h2o_process_automl['str_tag_data_experiment']
    df_output_info['tag_model_experiment'] = dict_h2o_process_automl['str_tag_model_experiment']
    df_output_info = df_output_info[['tag_data_experiment', 'tag_model_experiment', 'model_id', 'algo'] + [col for col in df_output_info.columns.tolist() if col != 'model_id' and col != 'algo' and col != 'tag_data_experiment' and col != 'tag_model_experiment']].copy()
    
    return df_output_info, pd.concat(list_df_feature_importance)
    
    
def ds_common_h2o_process_model_sub_get_info_stacked_ensemble(str_model_id):
    
    """
    TODO function & documentantion
    """
    
    return pd.DataFrame({'model_id': str_model_id}, index = [0]), pd.DataFrame({'model_id': str_model_id}, index = [0])


def ds_common_h2o_process_model_sub_get_info_deep_learning(str_model_id, model_h2o, list_hyperparams):
    
    """
    TODO documentantion
    """
    
    ## Extract model output
    dict_output = model_h2o._model_json["output"]
    
    ## Extract variable importance and add model_id
    df_feature_importance             = dict_output['variable_importances'].as_data_frame()
    df_feature_importance['model_id'] = str_model_id
    
    ## Remove features with 0 importance
    df_feature_importance            = df_feature_importance[df_feature_importance['percentage'] > 0].copy()
    
    ## Create feature column
    df_feature_importance['feature'] = df_feature_importance['variable'].map(lambda x: x.split(".")[0])
    
    ## Extract hyperparameters
    dict_out_hyper = {'model_id': str_model_id}
    for hyperparameter in list_hyperparams:
        if(type(model_h2o.params[hyperparameter]['actual'])==list):
            dict_out_hyper[hyperparameter] = ', '.join([str(x) for x in model_h2o.params[hyperparameter]['actual']])
        else:
            dict_out_hyper[hyperparameter] = model_h2o.params[hyperparameter]['actual']
    dict_out_hyper['number_variables'] = len(set(df_feature_importance['variable']))
    dict_out_hyper['number_features']  = len(set(df_feature_importance['feature']))
    
    return df_feature_importance, pd.DataFrame(dict_out_hyper, index = [0])


def ds_common_h2o_process_model_sub_get_info_gbm_xgb(str_model_id, model_h2o, list_hyperparams):
    
    """
    TODO documentantion
    """
    
    ## Extract model output
    dict_output = model_h2o._model_json["output"]
    
    ## Extract variable importance and add model_id
    df_feature_importance             = dict_output['variable_importances'].as_data_frame()
    df_feature_importance['model_id'] = str_model_id
    
    ## Remove features with 0 importance
    df_feature_importance            = df_feature_importance[df_feature_importance['percentage'] > 0].copy()
    
    ## Create feature column
    df_feature_importance['feature'] = df_feature_importance['variable'].map(lambda x: x.split(".")[0])
    
    ## Extract hyperparameters
    dict_out_hyper = {'model_id': str_model_id}
    for hyperparameter in list_hyperparams:
        if(type(model_h2o.params[hyperparameter]['actual'])==list):
            dict_out_hyper[hyperparameter] = ', '.join([str(x) for x in model_h2o.params[hyperparameter]['actual']])
        else:
            dict_out_hyper[hyperparameter] = model_h2o.params[hyperparameter]['actual']
    dict_out_hyper['number_variables'] = len(set(df_feature_importance['variable']))
    dict_out_hyper['number_features']  = len(set(df_feature_importance['feature']))
    
    return df_feature_importance, pd.DataFrame(dict_out_hyper, index = [0])


def ds_common_h2o_process_model_sub_get_info_glm(str_model_id, model_h2o, list_hyperparams):
    
    """
    TODO documentantion
    """
    
    ## Extract model output
    dict_output = model_h2o._model_json["output"]
    
    ## Extract variable importance and add model_id
    df_feature_importance             = dict_output['variable_importances'].as_data_frame()
    df_feature_importance['model_id'] = str_model_id
    
    ## Remove features with 0 importance
    df_feature_importance            = df_feature_importance[df_feature_importance['percentage'] > 0].copy()
    
    ## Create feature column
    df_feature_importance['feature'] = df_feature_importance['variable'].map(lambda x: x.split(".")[0])
    
    ## Extract hyperparameters (for GLM it is simpler to extract them in this way instead of using list_hyperparams)
    dict_out_hyper = {
                      'model_id':         str_model_id, 
                      'alpha':            dict_output['alpha_best'],
                      'lambda':           dict_output['lambda_best'],
                      'number_variables': len(set(df_feature_importance['variable'])),
                      'number_features':  len(set(df_feature_importance['feature']))
                     }
    
    # Remark: coefficient sign can be extracted from dict_output['coefficients_table'].as_data_frame()
    
    return df_feature_importance, pd.DataFrame(dict_out_hyper, index = [0])
