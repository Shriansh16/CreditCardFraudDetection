import pandas as pd
import numpy as np
import os
import sys
from sklearn.naive_bayes import GaussianNB
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
          
            gnb=GaussianNB()
          
            gnb.fit(X_train_arr,y_train_arr)
            y_pred=gnb.predict(X_test_arr)
            print("ACCURACY SCORE :",accuracy_score(y_test_arr,y_pred))
            save_object(self.model_trainer_config.model_path,gnb)
        except Exception as e:
            logging.info("ERROR OCCURED IN MODEL TRAINING")
            raise CustomException(e,sys)
