import os
import sys
import pandas as pd
sys.path.insert(0,'D:\CreditCardFraudDetection\src')
from logger import *
from exception import *
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    def initiate_DataIngestion(self):
        try:

            df=pd.read_csv(os.path.join("notebooks","CreditCardData.csv"))
            logging.info("DATA IS READ AS DATAFRAME")
            logging.info("DIVIDING DATA INTO TRAIN AND TEST DATASET")
            df.to_csv(self.data_ingestion_config.raw_data_path)
            train,test=train_test_split(df,test_size=0.20)
            train.to_csv(self.data_ingestion_config.train_data_path)
            test.to_csv(self.data_ingestion_config.test_data_path)
            return(self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path
                 
             )
        except Exception as e:
            logging.info("ERROR OCCURRED IN DATA INGESTION")
            raise CustomException(e,sys)

    