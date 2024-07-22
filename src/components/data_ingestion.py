import os 
import sys
sys.path.append(os.path.join(os.getcwd()))
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import *

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try : 
            df = pd.read_csv("notebook\data\data.csv")
            logging.info("Read the Dataset as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw Data saved in Workspace")

            logging.info("Train_Test_Split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state = 3)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Train and Test Data saved in Workspace")

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    try : 
        obj=DataIngestion()
        train_path,test_path = obj.initiate_data_ingestion()
        obj1 = DataTransformation()
        train_arr,test_arr,preprocessor_path = obj1.initiate_data_transformation(train_path,test_path)
        print(train_arr[0,:])
        print(train_arr.shape,test_arr.shape,preprocessor_path)
    except Exception as e : 
        raise CustomException(e,sys)