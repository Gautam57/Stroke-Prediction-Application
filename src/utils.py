import os
import sys
import dill
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            para=param[list(models.keys())[i]]

            logging.info(f"The model to be performed : {model} and its params passed : {para}")

            gs = GridSearchCV(model,para,cv=3, scoring='recall')
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model
            logging.info(f"Model fit completed")

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            logging.info(f"Model prediction completed")

            train_model_score = recall_score(y_train, y_train_pred)

            test_model_score = recall_score(y_test, y_test_pred)

            logging.info(f"The train score : {train_model_score} test score : {test_model_score}")

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys) 
    
def classimbalancetechnique(X_train,y_train,technique):
    try:
        if technique.upper() == "SMOTE":
            smote = SMOTE(sampling_strategy='auto', random_state=55)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    except Exception as e:
        raise CustomException(e, sys) 

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)