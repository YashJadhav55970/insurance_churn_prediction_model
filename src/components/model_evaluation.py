
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.constants import SCHEMA_FILE_PATH , TARGET_COLUMN
from src.utils.main_utils import read_yaml_file

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
       
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        

    def _drop_id_columns(self, df):
        logging.info("Dropping columns from schema config")

        drop_cols = self._schema_config.get("drop_columns", [])

        # keep only columns that actually exist
        cols_to_drop = [col for col in drop_cols if col in df.columns]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df
    

    def imputer_Num_cat(self, data: pd.DataFrame) -> pd.DataFrame:

    # -------- Numeric --------
        num_cols = data.select_dtypes(include=["number"]).columns.tolist()
        logging.info(f"Numeric columns: {num_cols}")

        num_imputer = SimpleImputer(strategy="median")
        num_df = pd.DataFrame(
            num_imputer.fit_transform(data[num_cols]),
            columns=num_cols,
            index=data.index
        )

    # -------- Categorical --------
        data = data.replace("?", np.nan)
        cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
        logging.info(f"Categorical columns: {cat_cols}")

        cat_imputer = SimpleImputer(strategy="constant", fill_value="UNKNOWN")
        cat_df = pd.DataFrame(
            cat_imputer.fit_transform(data[cat_cols]),
            columns=cat_cols,
            index=data.index
        )

        # -------- Combine --------
        final_df = pd.concat([num_df, cat_df], axis=1)

        return final_df
    

    def OHE_cat(self , data , encoder_col = None , encoder = None) ->pd.DataFrame:

        cat_cols = [col for col in data.columns if data[col].dtype == "object"]


        nominal = ['authorities_contacted', 'incident_state', 'insured_hobbies']

        data_ohe = data[nominal]

        logging.info("Aplying One Hot Encoding to Categorical Features")

        if encoder == None:
            encoder = OneHotEncoder(
                handle_unknown="ignore" ,
                drop = "if_binary"
            )

            encoder.fit(data_ohe)
            encoder_col = encoder.get_feature_names_out(data_ohe.columns)


        data_encoded = encoder.transform(data_ohe).toarray()
        data_encoded = pd.DataFrame(data_encoded ,
                                    index = data_ohe.index ,
                                    columns = encoder_col
                                    )
                
        data = data.drop(columns=nominal)
        data = pd.concat([data, data_encoded], axis=1)

        return data
    

    def OE_cat(self , data, encoder = None) -> pd.DataFrame:
        cat_cols = [col for col in data.columns if data[col].dtype == "object"]

        ordinal = ['collision_type', 'incident_type', 'incident_severity']

        data_le = data[ordinal]

        collision_type = ['UNKNOWN', 'Side Collision', 'Rear Collision', 'Front Collision']
        incident_severity = ['Trivial Damage','Minor Damage','Major Damage','Total Loss']
        incident_type = ['Parked Car','Single Vehicle Collision','Multi-vehicle Collision','Vehicle Theft']
        
        logging.info("Aplying Ordinal  Encoding to Categorical Features")

        if encoder == None:
            # Create object
            encoder = OrdinalEncoder(categories=[collision_type, incident_type,incident_severity])
            encoder.fit(data_le)

        ## Transform the data
        data_encoded = encoder.transform(data_le)
        data_encoded = pd.DataFrame(data_encoded,
                                    index = data_le.index,
                                    columns = data_le.columns)


        # Concatenating categorical feature after applying Oridinal Encoding
        data = data.drop(columns=ordinal)
        data = pd.concat([data, data_encoded], axis=1)

        return data
    

    def label_encoding(self , data):

        # cat_cols = [col for col in data.columns if data[col].dtype == "object"]
        cat_cols = data.select_dtypes(include=["object"]).columns.tolist()

        logging.info("Applying Label  Encoding to Categorical Features")

        for cat in cat_cols:

            le = LabelEncoder()

            data[cat] = le.fit_transform(data[cat])

        return data

        


    def evaluate_model(self) -> EvaluateModelResponse:
       
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")


            x = self._drop_id_columns(x)
            x = self.imputer_Num_cat(x)
            x = self.OHE_cat(x)
            x = self.OE_cat(x)
            x = self.label_encoding(x)

            y = y.map({"N": 0, "Y": 1})

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")
            print(f"--------------------------------------> {type(trained_model_f1_score)}")


            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result
            '''
            '''

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,

                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
                
                )

            '''
            '''
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
