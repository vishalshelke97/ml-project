import os
import sys
from src.excepetion import CustomExpection
import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomExpection(e, sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models,param):
    try:
       report = {}
       for i in range(len(list(models))):
           model = list(models.values())[i]
           para= param[list(models.keys())[i]]

           gs=GridSearchCV(model, param_grid=para, cv=3)
           gs.fit(x_train, y_train) # Train model

           model.set_params(**gs.best_params_) # Set best parameters
           model.fit(x_train, y_train) # Train model

          # model.fit(x_train, y_train) # Train model
           y_train_pred = model.predict(x_train) # Predict on training data
           y_test_pred = model.predict(x_test) # Predict on test data

           train_model_score = r2_score(y_train, y_train_pred) # Evaluate model on training data
           test_model_score = r2_score(y_test, y_test_pred) # Evaluate model on test data
           
           report[list(models.keys())[i]] = test_model_score # Store test model score in report

       return report
    
          
    except Exception as e:
        raise CustomExpection(e, sys)
        

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomExpection(e, sys)