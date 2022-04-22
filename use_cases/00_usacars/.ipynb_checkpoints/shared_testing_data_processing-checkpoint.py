## Import libraries
import numpy as np
import pandas as pd
import logging
from shared_core_data_processing import * 


## Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


## Define unit test functions
def run_unit_test_00_ds_common_pandas_dataframe_split_main():
    
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
