import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artificats', 'train.csv')
    test_data_path: str= os.path.join('artificats', 'test.csv')
    full_data_path: str= os.path.join('artificats', 'df.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self):
        logging.info("Staring to load the dataset")
        try:
            df = pd.read_csv('/Users/abhi/Desktop/End to End ML/Notebook[EDA-Model]/Data/predictive_maintenance.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.full_data_path, index=False, header=True)

            logging.info("Initiated training and testing splitting")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train Test Split executed successfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    obj.start_data_ingestion()
        


            


