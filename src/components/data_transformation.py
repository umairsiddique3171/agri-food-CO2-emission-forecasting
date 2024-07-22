import os 
import sys
sys.path.append(os.path.join(os.getcwd()))
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_selected_features
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.p')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_features = selector(dtype_exclude="object")
            cat_features = selector(dtype_include="object")
            logging.info(f"Categorical Columns : {cat_features}")
            logging.info(f"Numerical Columns : {num_features}")

            numeric_transformer = Pipeline(
                steps = [
                    ('scaler', StandardScaler())
                ])
            
            categorical_transformer = Pipeline(
                steps = [
                    ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])
        
            preprocessor = ColumnTransformer(
                transformers = [
                    ('num', numeric_transformer, num_features),
                    ('cat', categorical_transformer, cat_features)
                    ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read Train and Test Data for Data Transformation")

            preprocessing_obj = self.get_data_transformer_obj()
            logging.info("Obtained Preprocessing Object")

            train_df = train_df.dropna()
            test_df = test_df.dropna()
            logging.info("Null Values Dropped")

            selected_features_path = "notebook\selected_features.json"
            selected_features = load_selected_features(selected_features_path)
            logging.info(f"Number of Selected Features : {len(selected_features)}")

            target_column_name = 'total_emission'
            input_feature_train_df = train_df[selected_features]
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df[selected_features]
            target_feature_test_df=test_df[target_column_name]
            logging.info("Features_Target_Split Done")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Numerical Columns Scaled")
            logging.info("Categorical Columns Encoded")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            logging.info("Preprocessor Object Saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)