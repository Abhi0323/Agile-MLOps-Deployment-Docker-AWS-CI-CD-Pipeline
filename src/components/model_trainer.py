import os
import sys

from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelConfig:
    trained_model_path = os.path.join("artificats", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelConfig()


    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Spliting training and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "KNN": KNeighborsClassifier()
            }

            params = {
                        "Random Forest": {
                            'n_estimators': [10, 50, 100],
                            'max_depth': [None, 10, 20]
                    },
                        "SVM": {
                            'C': [0.1, 1, 10],
                            'kernel': ['linear', 'rbf']
                    },
                        "Gradient Boosting": {
                            'learning_rate': [0.1, 0.01],
                            'n_estimators': [50, 100]
                    },
                        "Logistic Regression": {
                            'C': [0.1, 1, 10],
                            'solver': ['liblinear', 'saga']
                    },
                        "KNN": {
                            'n_neighbors': [3, 5, 10],
                            'weights': ['uniform', 'distance']
                    }
                    }


            Per_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            best_model_score = max(sorted(Per_report.values()))

            ## To get best model name from dict

            best_model_name = list(Per_report.keys())[
                list(Per_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best found model on both train and test dataset")

            save_object(
                file_path=self.model_config.trained_model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)
