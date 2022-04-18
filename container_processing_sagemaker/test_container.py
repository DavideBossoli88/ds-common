
import h2o
import logging
import argparse

if __name__ == '__main__':
    
    # Parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter1')
    
    args, _ = parser.parse_known_args()
    
    parameter1 = args.parameter1
    
    print(parameter1)
    
    # Setup H2O
    h2o.init()
