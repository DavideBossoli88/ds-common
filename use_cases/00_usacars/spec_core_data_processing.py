## Import libraries
import numpy as np
import pandas as pd
import logging


## Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


## Define core functions
def extract_expiring_minutes(x):
    
    """
    This function takes as input a string formatted as '<number> <metric> <xx>' and returns the minutes before expiring.
    When x = 'Listing expired' we set it to 0.
    When x is null it returns -999. General errors are handled through a -99. 
    
    Parameters
    ----------
    x : string
    
    Returns
    ----------
    float 
    
    
    Remarks
    ----------
    I decided to return a float instead of an int, because I might have used H2O instead of Sklearn, and it is generally safer using floats for H2O.
    """
    
    # If input is null, return -999
    if pd.isnull(x):
        return float(-999)
    
    # Make string to lower to make it safer
    x = x.lower()
    
    if(x=='listing expired'):
        return float(0)
    else:
        try:
            lst_split        = x.split(" ")
            str_split_number = lst_split[0]
            str_split_metric = lst_split[1]
            
            if(str_split_metric == 'minutes'):
                return float(str_split_number)
            elif(str_split_metric == 'hours'):
                return float(str_split_number)*60
            elif(str_split_metric == 'days'):
                return float(str_split_number)*60*24
            else:
                return float(-999)
        except:
            return float(-99)



## Define unit test functions
def run_unit_tests_extract_expiring_minutes(dict_unit_tests):
    
    """
    This function runs unit tests for the function extract_expiring_minutes(x).
    
    Parameters
    ----------
    dict_unit_tests : dictionary key-value 
                      key   is the input to be tested
                      value is the expected output
    
    """
    
    logging.info(' ')
    logging.info('########################')
    logging.info('Start unit tests for function extract_expiring_minutes')
    
    count_errors = 0
    for str_input in dict_unit_tests.keys():
        str_output = extract_expiring_minutes(str_input)
        if(str_output != dict_unit_tests[str_input]):
            count_errors = count_errors + 1
            logging.info(f'Input: {str_input} -> output: {str_output}')
        
    logging.info('########################')
    logging.info(f'End unit tests for function extract_expiring_minutes. Number of errors: {count_errors}')
    logging.info(' ')
    
    return