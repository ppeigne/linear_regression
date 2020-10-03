import pandas as pd
import numpy as np

def check_args_predict_csv(argv):
    size = len(argv)
    if size == 0 :
        print("Error! Argument mising.\nPlease supply a valid path to the dataset.")
        exit()
    elif size > 1:
        print("Warning! Too many arguments!\nOnly the first argument will be considered.")

def check_args_predict(argv):
    size = len(argv)
    if size == 0 :
        print("Error! Argument mising.\nPlease supply data.")
        exit()

def check_args_train(argv):
    size = len(argv)
    if size == 1:
        print("Error! Argument mising.\nPlease supply a valid path to the dataset.")
        exit()
    valid_args = {'model' : ['linear', 'ridge'], 'scaler' : ['standard', 'minmax'], 'polynomial' : [str(i) for i in range(1,10)]}
    args = {}
    if size == 2:
        default_args = [p[0] for p in valid_args.values()]
        print(f'No arguments given. The program will use the default parameters: {default_args}.')
    else :
        for i in range(size - 2):
            key = list(valid_args.keys())[i]
            if argv[i + 2] not in valid_args[key]:
                print(f"Error! Invalid argument.\nPlease supply a valid {key} argument: {valid_args[key]}, or remove it to use default parameters.")
                exit()
            else: 
                args[key] = argv[i+2]
    return args           

def get_data(data_path):
    try:
        data = pd.read_csv(data_path)
    except:
        print("Error! Invalid argument. Please supply a valid path to the dataset.")
        exit()
    return data