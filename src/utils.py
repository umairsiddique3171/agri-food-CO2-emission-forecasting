import pickle
import sys
import os
import json
import base64

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException


def save_object(file_path,obj):

    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    except Exception as e: 
        raise CustomException(e,sys)
    

def load_object(path):
    
    try : 
        model = pickle.load(open(path, "rb"))
        return model
    
    except Exception as e: 
        raise CustomException(e,sys)


def load_selected_features(file_path):

    try:
        with open(file_path,'r') as file_obj:
            features_list = json.load(file_obj)
        return features_list
    
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train,y_train,X_test,y_test,models):

    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)  
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = (train_model_score,test_model_score)
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models_with_tuning(X_train,y_train,X_test,y_test,models,param):

    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = (train_model_score,test_model_score)
        logging.info("Best Parameters for each Model are selected after Tuning")
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    

def save_report(report,path):

    try: 
        models_training_report = pd.DataFrame(data={
                'models' : report.keys(),
                'training_r2' : [v[0] for v in list(report.values())],
                'testing_r2' : [v[1] for v in list(report.values())]
            })
        models_training_report.to_csv(path,header=True,index=False)
        
    except Exception as e: 
        raise CustomException(e,sys)
    

def set_background(img_file):

    with open(img_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)