import os
import sys
import pickle
sys.path.insert(0,'D:\CreditCardFraudDetection\src')
from logger import *
from exception import *

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        logging.info("ERROR OCCURED IN SAVING THE PICKLE FILE")
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("ERROR OCCURED IN LOADING THE OBJECT")
        raise CustomException(e,sys)
         