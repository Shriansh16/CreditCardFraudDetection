import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import os
import sys
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
sys.path.insert(0,'D:\CreditCardFraudDetection\src')
from logger import *
from exception import *
from utils import *

@dataclass
class ModelTrainerConfig:
    model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_training(self,train_arr,test_arr):
        try:
            
            X_train_arr,y_train_arr,X_test_arr,y_test_arr = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            '''xgb=XGBClassifier()
            param_grid_xgboost = { "n_estimators": [50,100, 130],
                        "max_depth": range(3,11,1),
                      "random_state":[0,50,100]
                     }
            grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),param_grid_xgboost,verbose=3,cv=5,n_jobs=-1)
            grid.fit(X_train_arr,y_train_arr)
            y_pred=grid.predict(X_test_arr)
            print("training completed")
            logging.info("training completed")
            '''
            rvc=RandomForestClassifier()
            rvc.fit(X_train_arr,y_train_arr)
            y_pred=rvc.predict(X_test_arr)
            print("ACCURACY SCORE :",accuracy_score(y_test_arr,y_pred))
            save_object(self.model_trainer_config.model_path,rvc)
        except Exception as e:
            logging.info("ERROR OCCURED IN MODEL TRAINING")
            raise CustomException(e,sys)
