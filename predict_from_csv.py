import numpy as np
import pandas as pd
import joblib
import sys
from utils import check_args_predict_csv, get_data
from linear_model import LinearRegression_
import matplotlib.pyplot as plt

check_args_predict_csv(sys.argv[1:])
data = get_data(sys.argv[1])
try:
    model = joblib.load('model.pkl')
except:
    model = LinearRegression_()

results = model.predict(data)
predictions = pd.concat((data, pd.DataFrame(results, columns=['predictions'])), axis=1)
print('Results:')
print(predictions)

i1 = input("Save results? [y,n]\n")
if i1.lower() in ['y', 'yes']:
    predictions.to_csv('results.csv', index=False)
    print("Results saved as results.csv")