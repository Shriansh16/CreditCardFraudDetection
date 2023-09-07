import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'D:\CreditCardFraudDetection\src')
from logger import *
from exception import *
from utils import *
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_file_obj_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            preprocess=StandardScaler()
            return preprocess
        except Exception as e:
            logging.info("ERROR OCCURED IN PREPROCESSOR")
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("DATA TRANSFORMATION INITIATED")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            train_independent_data=train_df.drop(columns='default payment next month',axis=1)
            train_dependent_data=train_df['default payment next month']
            test_independent_data=test_df.drop(columns='default payment next month',axis=1)
            test_dependent_data=test_df['default payment next month']
            preprocessor=self.get_data_transformation_object()
            transformed_input_train=preprocessor.fit_transform(train_independent_data)
            transformed_input_test=preprocessor.transform(test_independent_data)
            train_arr=np.c_[transformed_input_train,np.array(train_dependent_data)]
            test_arr=np.c_[transformed_input_test,np.array(test_dependent_data)]
            save_object(self.data_transformation_config.preprocessor_file_obj_path,preprocessor)
            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_file_obj_path
            )
        except Exception as e:
            logging.info("ERROR OCCURED DURING DATA TRANSFORMATION")
            raise CustomException(e,sys)


