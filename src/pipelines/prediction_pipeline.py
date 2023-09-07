import pandas as pd
import os
import sys
from flask import request
sys.path.insert(0,'D:\CreditCardFraudDetection\src')
from logger import *
from exception import *
from utils import *
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_file_path=os.path.join("predictions","prediction_file.csv")
    model_file_path=os.path.join("artifacts","model.pkl")
    preprocessor_file_path=os.path.join("artifacts","preprocessor.pkl")
class PredictionPipeline:
    def __init__(self,request:request):
        self.request=request
        self.prediction_pipeline_config=PredictionPipelineConfig()
    def save_input_files(self):
        try:

            input_file_name="input_files"
            os.makedirs(input_file_name,exist_ok=True)
            input_csv_file=self.request.files['file']
            input_file_path=os.path.join(input_file_name,input_csv_file.filename)
            input_csv_file.save(input_file_path)
            return input_file_path
        except Exception as e:
            logging.info("ERROR OCCURED IN SAVING THE INPUT FILE")
            raise CustomException(e,sys)
    def predict(self,features):
        try:
            model=load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor=load_object(self.prediction_pipeline_config.preprocessor_file_path)
            transformed_features=preprocessor.fit_transform(features)
            predictions=model.predict(transformed_features)
            return predictions
        except Exception as e:
            logging.info("ERROR OCCURED DURING PREDICTION")
            raise CustomException(e,sys)
    def save_prediction_file(self,input_dataframe_path):
        try:
            prediction_column_name="TARGET_COLUMN"
            input_file=pd.read_csv(input_dataframe_path)
            preds=self.predict(input_file)
            input_file[prediction_column_name]=[pred for pred in preds]
            target_column_mapping={0:"fraud",1:"Not a fraud"}
            input_file[prediction_column_name]=input_file[prediction_column_name].map(target_column_mapping)
            input_file.to_csv(self.prediction_pipeline_config.prediction_file_path,index=False)
            logging.info("PREDICTIONS COMPLETED")
        except Exception as e:
            logging.info("ERROR OCCURED IN SAVING THE PREDICTION FILE")
            raise CustomException(e,sys)
    def run_pipeline(self):
        try:
            input_csv_path=self.save_input_files()
            self.save_prediction_file(input_csv_path)
            return self.prediction_pipeline_config
        except Exception as e:
            logging.info("ERROR OCCURED IN RUNNIG THE PREDICTION PIPELINES")
            raise CustomException(e,sys)


