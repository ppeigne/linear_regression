import numpy as np
import pandas as pd
import joblib
import sys
from utils import check_args_train, get_data
from selector import Selector, Pipeline

args = check_args_train(sys.argv)
data = get_data(sys.argv[1])

slct = Selector(**args)
pip = slct.build_pipeline()

X = data[data.columns[:-1]]
y = data[data.columns[-1]]

print("Feeding data to the model's pipeline.")
pip.fit(X,y)
print("Model trained.")
joblib.dump(pip, "model.pkl")
print("Model saved as model.pkl")