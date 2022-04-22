## Import libraries
import numpy as np
import pandas as pd
import logging


## Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


## Define core functions
def ds_common_pandas_dataframe_split_main(df, str_colname_split, dict_splits, dict_metrics, dict_metrics_tolerance, int_seed, int_max_number_iterations):
    
    """
    This function returns a dataframe with an extra column called <str_colname_split> created with a custom dataframe split function.
    
    The main custom functionality are:
    1. specifying the split column name.
    2. specifying the split column levels.
    3. specifying a set of conditions required for the split to be used.
    These conditions are specified with two parameters:
    - dict_metrics           -> pandas-like group by aggregation dictionary
    - dict_metrics_tolerance -> key-value dictionary containing the tolerance of each metric
    
    The tolerance is related to the following function:
    max(kpi over all the splits) - min(kpi over all the splits) - 1
    
    
    Parameters
    ----------
    df : pandas Dataframe
    
    str_colname_split : string
                        The name of the column that will be created as train/test/validation fold
              
    dict_splits : dictionary key-value
                  Key   = name of the level
                  Value = proportion of sample
                 
    dict_metrics : dictionary key-value
                   Standard dictionary used for pandas group by aggregated functions  
                   
    dict_metrics_tolerance : dictionary key-value
                             Contains the tolerance of the computed metric max/min - 1
                    
    int_seed : integer
               The first iteration will start from this seed number, afterwards it will increase it by 1
    
    int_max_number_iterations : integer
                                Number of iterations to be tested. If convergency is not achieved, the last iteration will be used.
    
    
    Returns
    ----------
    df
    """

    ## Check input correctness
    # The sum of values of dict_splits must be 1
    sum_values = 0
    for key in dict_splits.keys():
        sum_values = sum_values + dict_splits[key]
    if(sum_values != 1):
        raise Exception("The sum of values of dict_splits must be 1")
        
    # All keys of dict_metrics must be available in df
    if (len(set(dict_metrics.keys()) - set(df.columns.tolist())) !=0):
        raise Exception("All keys of dict_metrics must be available in df")   
    
    
    ## Iteratively look for a correct split
    count_errors     = 999
    count_iterations = 1
    while count_errors > 0:
        
        ## Create str_colname_split
        df[str_colname_split] = ds_common_pandas_dataframe_split_sub_get_str_colname_split(df, int_seed, dict_splits)
            
        ## Get dict_tolerance_errors
        dict_tolerance_errors, dict_computed_metrics, df_metrics =  ds_common_pandas_dataframe_split_sub_get_dict_tolerance_errors(df, dict_metrics, dict_metrics_tolerance, str_colname_split)
        
        ## If there are no errors, convergency is achieved: exit the loop.
        if(len(dict_tolerance_errors) == 0):
            count_errors = 0
            logging.info(f'Achieved metrics convergence at seed {int_seed} and iteration {count_iterations}')
        ## Otherwise, update the seed counter and restart the loop.
        else:
            int_seed         = int_seed + 1
            count_iterations = count_iterations + 1
            ## If the maximum number of iteration is reached, exit the loop and throw a warning.
            if(count_iterations > int_max_number_iterations):
                logging.info(f'Warning: metrics did not converge, last iteration metrics are {dict_tolerance_errors}')
                logging.info('Using the last iteration split')
                break
    return df


def ds_common_pandas_dataframe_split_sub_get_str_colname_split(df, int_seed, dict_splits):
    
    """
    This function computes a list of strings corresponding to the levels of the split column to be created.
    
    
    Parameters
    ----------
    df : pandas Dataframe
    
    int_seed : integer
                  The first iteration will start from this seed number, afterwards it will increase it by 1
              
    dict_splits : dictionary key-value
                  Key   = name of the level
                  Value = proportion of sample
                    

    Returns
    ----------
    list 
    """
    
    ## Set the random seed
    np.random.seed(int_seed)
    
    
    ## Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    
    ## Sample from a multinomial distribution
    array_distribution = np.random.multinomial(1, [value for value in dict_splits.values()], size=df.shape[0])
    
    
    ## Create str_colname_split
    list_str_colname_split = []
    for i in range(0, len(array_distribution - 1)):
        cont = 0
        for str_key in dict_splits.keys():
            if(array_distribution[i][cont] == 1):
                list_str_colname_split.append(str_key)
            else:
                pass
            cont = cont + 1
            
    return list_str_colname_split

def ds_common_pandas_dataframe_split_sub_get_dict_tolerance_errors(df, dict_metrics, dict_metrics_tolerance, str_colname_split):
    
    """
    This function computes a list of strings corresponding to the levels of the split column to be created.
    
    
    Parameters
    ----------
    df : pandas Dataframe
                 
    dict_metrics : dictionary key-value
                   Standard dictionary used for pandas group by aggregated functions  
                   
                   
    dict_metrics_tolerance : dictionary key-value
                         Contains the tolerance of the computed metric max/min - 1
                   
    str_colname_split : string
                        The name of the column that will be created as train/test/validation fold 


    Returns
    ----------
    dict_tolerance_errors : dictionary
    
    dict_computed_metrics : dictionary
    
    df_metrics : pandas frame
    """
    
    ## Compute metrics
    df_metrics         = df[[str_colname_split] + [key for key in dict_metrics.keys()]].groupby(by=str_colname_split).agg(dict_metrics).reset_index()
    df_metrics.columns = df_metrics.columns.get_level_values(0) + '_' +  df_metrics.columns.get_level_values(1)
    
    dict_computed_metrics = {}
    for str_column in df_metrics.columns.tolist():
        if(str_colname_split not in str_column):
            dict_computed_metrics[str_column] =  df_metrics[str_column].max() / df_metrics[str_column].min() - 1
            
    ## Verify metrics tolerance
    dict_tolerance_errors = {}
    for str_key in dict_computed_metrics.keys():
        if(dict_computed_metrics[str_key] > dict_metrics_tolerance[str_key]):
            dict_tolerance_errors[str_key] = [dict_computed_metrics[str_key], dict_metrics_tolerance[str_key]]
            
    return dict_tolerance_errors, dict_computed_metrics, df_metrics


## Define unit test functions
def unit_test_ds_common_pandas_dataframe_split_main():
    
    # Get dummy test_df
    test_df = pd.DataFrame(
                           {
                            'price':   [i for i in range(0, 10000)], 
                            'data_id': [str(i) for i in range(0, 10000)]
                           }
                          )
    
    # Setup parameters
    str_colname_split         = 'flag_type'
    dict_splits               = {'train': 0.5, 'test': 0.25, 'validation': 0.25}
    dict_metrics              = {'price': ['mean', 'std']}
    dict_metrics_tolerance    = {'price_mean': 0.05, 'price_std': 0.05}
    int_max_number_iterations = 100
    int_seed                  = 1 
    
    return ds_common_pandas_dataframe_split_main(test_df, str_colname_split, dict_splits, dict_metrics, dict_metrics_tolerance, int_seed, int_max_number_iterations)


def ds_common_get_list_constant_features_main(df, lst_features):
    
    """
    This function returns a list of constant features of a dataframe.
    
    Parameters
    ----------
    df : pandas Dataframe
    
    lst_features : list
                   List of columns to be analyzed
    
    Returns
    ----------
    list of constant columns 
    """
    
    lst_constant_features = []
    for str_col in lst_features:
        if (df[str_col].value_counts().shape[0] == 1):
            lst_constant_features.append(str_col)
        else:
            pass
            
    logging.info(f'Df has the following constant features: {lst_constant_features}')
    
    return lst_constant_features
