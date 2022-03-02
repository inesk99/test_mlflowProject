# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:14:27 2022

@author: Inès
"""

# librairy 
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

import os 
import sys

#os.chdir("C:/Users/Inès/Cours M2 SISE/Atelier - Ricco R/Mlflow/Project-test2")

data_path = "iris.csv"

def eval_metrics (actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    
    return rmse,mae, r2

def load_data(): 
    data = load_diabetes()
    
    X = data.data
    y = data.target
    
    X = pd.DataFrame(X,columns=data.feature_names)
    y = pd.DataFrame(y,columns=["target"])
    
    train_x,test_x, train_y,test_y = train_test_split(X,y, train_size=0.7)
    
        
    return train_x,test_x, train_y,test_y

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    
if __name__ == "__main__":
    
    
    X_train, X_test, y_train, y_test = load_data()
    
    
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=ratio)
        lr.fit(X_train,y_train)
        
        y_pred = lr.predict(X_test)
        (rmse,mae,r2) = eval_metrics(y_test, y_pred)
        
        print("rmse %s" % rmse)
        print("mae %s" % mae)
        print("r2 %s" %r2)
        
        
        mlflow.log_params({"alpha":alpha,"l1_ratio":ratio})
        mlflow.log_metrics({"rmse":rmse,"mae":mae,"r2":r2})
        #mlflow.log_artifact()
        
        mlflow.sklearn.autolog()

