## Import libraries
import numpy as np
import pandas as pd
import logging
from spec_core_data_processing import *


## Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def run_unit_test_00_extract_expiring_minutes():

    ## Define a batch of unit test to run
    dict_ut_extract_expiring_minutes = {
                                    '3 hours left':              3*60,
                                    '3 minutes left':               3,
                                    '3 days left':            3*24*60,
                                    'listing expired':              0,
                                    '3 Hours left':              3*60,
                                    '3 Minutes left':               3,
                                    '3 Days left':            3*24*60,
                                    'Listing expired':              0,
                                    '':                           -99,
                                    ' ':                         -999,
                                    'this should return a -999': -999,
                                    'thisshouldreturna-99':       -99
                                   }
    
    run_batch_unit_tests_extract_expiring_minutes(dict_ut_extract_expiring_minutes)
    
    return


## Define unit test functions
def run_batch_unit_tests_extract_expiring_minutes(dict_unit_tests):
    
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