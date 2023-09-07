import pandas as pd
import os
import sys
from pathlib import Path
sys.path.insert(0,'D:\CreditCardFraudDetection\src')
from logger import *
from exception import *
from utils import *
sys.path.insert(0, 'D:\CreditCardFraudDetection\src\components')
from data_ingestion import *
from data_transformation import *
from model_trainer import *

if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_DataIngestion()
    obj1=DataTransformation()
    train_data,test_data,_=obj1.initiate_data_transformation(train_path,test_path)
    obj2=ModelTrainer()
    obj2.initiate_model_training(train_data,test_data)
