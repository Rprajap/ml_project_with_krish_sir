import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifact",'processor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_collumn = ['reading_score', 'writing_score']
            categorial_collumn = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
               ]
               )
            cat_pipline = Pipeline(
                steps=[

                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
                ])
            logging.info("Numerical  collumn Standared scalling completed")
            logging.info("Categorical collumn encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ('num_pipline',num_pipline,numerical_collumn),
                    ('cat_pipline',cat_pipline,categorial_collumn)

                ]

            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train  and test data is completed ')
            logging.info("obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformation_obj()

            target_column_name = "math_score"
            numerical_columns = ['reading_score', 'writing_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on train and test dataset")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('saved processing object.')


            save_object(
                
                file_path =self.data_tranformation_config.preprocessor_obj_file_path ,
                obj = preprocessor_obj
                        )

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        


        

        
 