import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass

@dataclass
class DataTransformetaionConfig:
    data_processor_obj_file_path= os.path.join("artificats", "processor.pkl")

class DataTransformetaion:
    def __init__(self):
        self.data_transformetaion_config = DataTransformetaionConfig()

    def get_transformed_data(self):
        try:
            ''' This function is mainly responible for transforing data '''
            num_features = ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
            cat_features = ["Type"]

            num_pipeline = Pipeline([("scaler", StandardScaler())])
            
            cat_pipeline = Pipeline(steps=[
                ("onehot", OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
                ])
            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features)
                ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_transfromation(self, train_path, test_path):
        try: 
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data read successfully")

            logging.info("Obtaining preprocessing object")

            preprocess_obj = self.get_transformed_data()

            target_variable = ['Target']
            not_imp_variables = ['UDI' , 'Product ID', 'Failure Type']

            input_features_train_df = train_df.drop(columns= target_variable + not_imp_variables, axis=1)
            target_train_df = train_df[target_variable]

            input_features_test_df = test_df.drop(columns= target_variable + not_imp_variables, axis=1)
            target_test_df = test_df[target_variable]

            logging.info(
                    f"Applying preprocessing obj on train df and test df"
                )
            input_features_train_df_array = preprocess_obj.fit_transform(input_features_train_df)
            input_features_test_df_array = preprocess_obj.transform(input_features_test_df)

            logging.info(
                    f"Applied preprocessing obj on train df and test df"
                )
            
            train_array = np.c_[input_features_train_df_array, np.array(target_train_df)]
            test_array = np.c_[input_features_test_df_array, np.array(target_test_df)]
            
            logging.info(
                    f"Saving preprocessing object"
                )
            
            save_object(file_path=self.data_transformetaion_config.data_processor_obj_file_path, obj= preprocess_obj)

            logging.info(
                    f"Saved preprocessing object"
                )
            
            return (
                train_array, 
                test_array, 
                self.data_transformetaion_config.data_processor_obj_file_path
                )
        except Exception as e:
            raise CustomException(e,sys)
            

