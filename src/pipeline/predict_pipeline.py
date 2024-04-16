import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class Pred_Pipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artificats', 'model.pkl')
            processor_path = os.path.join('artificats', 'processor.pkl')

            model = load_object(file_path=model_path)
            transformer = load_object(file_path=processor_path)

            data_trans = transformer.transform(features)
            output = model.predict(data_trans)
            return output
        except Exception as e:
            raise CustomException(e, sys)
        
class input_data:
    def __init__(self,
                 Type: str,
                 Air_temperature: float,
                 Process_temperature: float,
                 Rotational_speed: int,
                 Torque: float,
                 Tool_wear: int
    ):
        self.Type = Type
        self.Air_temperature = Air_temperature
        self.Process_temperature = Process_temperature
        self.Rotational_speed = Rotational_speed
        self.Torque = Torque
        self.Tool_wear = Tool_wear

    def transfrom_data_as_dataframe(self):
        try:
            user_input_data_dict= {
                "Type": [self.Type],
                "Air temperature [K]": [self.Air_temperature],
                "Process temperature [K]": [self.Process_temperature],
                "Rotational speed [rpm]": [self.Rotational_speed],
                "Torque [Nm]": [self.Torque],
                "Tool wear [min]": [self.Tool_wear]
            }
            return pd.DataFrame(user_input_data_dict)
        except Exception as e:
            raise CustomException(e, sys)


                 
    
