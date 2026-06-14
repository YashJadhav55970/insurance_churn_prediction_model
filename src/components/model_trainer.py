import sys
from typing import Tuple

import numpy as np
from lightgbm import LGBMClassifier

import yaml


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self , data_transformation_artifact:DataTransformationArtifact , 
                 model_trainer_config:ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config




    def get_model_obect_and_report(self , train:np.array , test:np.array) -> Tuple[object , object]:

        try:
            logging.info("Training RandomForestClassifier with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            # Initialize LightGBM with specified parameters

            # params = config['model_params']['LGBMClassifier']

            model = LGBMClassifier(

                boosting_type=self.model_trainer_config._model_boosting_type,
                # class_weight=self.model_trainer_config._model_class_weight,
                colsample_bytree=self.model_trainer_config._model_colsample_bytree,
                learning_rate=self.model_trainer_config._model_learning_rate,
                max_depth=self.model_trainer_config._model_max_depth,
                min_child_samples=self.model_trainer_config._model_min_child_samples,
                min_child_weight=self.model_trainer_config._model_min_child_weight,
                min_split_gain=self.model_trainer_config._model_min_split_gain,
                n_estimators=self.model_trainer_config._model_n_estimator,
                n_jobs=self.model_trainer_config._model_n_jobs,
                num_leaves=self.model_trainer_config._model_num_leaves,
                objective=self.model_trainer_config._model_model_objective,
                random_state=self.model_trainer_config._model_random_state,
                reg_alpha=self.model_trainer_config._model_reg_alpha,
                reg_lambda=self.model_trainer_config._model_reg_lambda,
                subsample=self.model_trainer_config._model_subsample,
                subsample_for_bin=self.model_trainer_config._model_subsample_for_bin,
                subsample_freq=self.model_trainer_config._model_n_estimator
            )


            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Prediction and evaluation metrics
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test , y_pred)
            f1 = f1_score(y_test , y_pred)
            precision = precision_score(y_test , y_pred)
            recall = recall_score(y_test , y_pred)

            metric_artifact = ClassificationMetricArtifact(accuracy=accuracy , f1_score=f1 , precision_score=precision ,recall_score=recall)

            return model , metric_artifact
        


        except Exception as e:
            raise MyException(e,sys)
        


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_obect_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e