import sys
import os
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder , OrdinalEncoder , LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN , SCHEMA_FILE_PATH , CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact , DataIngestionArtifact , DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_numpy_array_data , save_object , read_yaml_file





class DataTransformations:

    def __init__(self , data_ingestion_artifact :DataIngestionArtifact , 
                 data_transformation_config:DataTransformationArtifact ,
                 data_validation_artifact:DataValidationArtifact
                 ):
        

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e,sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    '''
    def get_data_transformer_object(self) -> Pipeline:

    
        logging.info("Entered get_data_transformer_object method of DataTransformation class")



        try:    
            numeric_transformer = StandardScaler()
            logging.info("Tranformation started - StandardScaler")

            num_features = self._schema_config["num_feature"]
            logging.info("Columns loaded from schema")

            # prepricessor pipline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardSclaer" , numeric_transformer , num_features)
                ],
                remainder="passthrough"
            )

            final_pipline = Pipeline(steps=[("Preprocessor" , preprocessor)])
            logging.info("Final Pipline Ready!!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipline
        

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e,sys)

        
    '''


    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Building the preprocessor pipeline...")
            num_features = self._schema_config["num_feature"]
            
            # Categorical columns management
            oh_columns = ['authorities_contacted', 'incident_state', 'insured_hobbies', 'insured_occupation']
            ordinal_columns = ['collision_type', 'incident_type', 'incident_severity']
            
            # Exact categories for Ordinal columns
            collision_list = ['UNKNOWN', 'Side Collision', 'Rear Collision', 'Front Collision']
            severity_list = ['Trivial Damage','Minor Damage','Major Damage','Total Loss']
            type_list = ['Parked Car','Single Vehicle Collision','Multi-vehicle Collision','Vehicle Theft']

            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", StandardScaler(), num_features),
                    ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"), oh_columns),
                    ("OrdinalEncoder", OrdinalEncoder(
                        categories=[collision_list, type_list, severity_list],
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    ), ordinal_columns)
                ],
                remainder="drop" # Safest option for ML pipelines
            )

            return Pipeline(steps=[("Preprocessor", preprocessor)])

        except Exception as e:
            raise MyException(e, sys)








    '''
    def _drop_id_columns(self , df):
        # Drop the id and _c39 column
        logging.info("Dropping 'id' and '_c39' columns")
        drop_col = self._schema_config["drop_columns"]
        if drop_col in df.columns:
            df = df.drop(drop_col)
        
        return df
    '''

    def _drop_id_columns(self, df):
        logging.info("Dropping columns from schema config")

        drop_cols = self._schema_config.get("drop_columns", [])

        # keep only columns that actually exist
        cols_to_drop = [col for col in drop_cols if col in df.columns]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        '''
        # Testing only 
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Dtypes:\n", df.dtypes)

        '''
        return df

  








    '''
    def imputer_Num_cat(self , data, imputer=None):


        # data.drop(columns = "_c39" , inplace = True)

        num_cols = [col for col in data.columns if data[col].dtype in ['int64', 'int32', 'float64']]

        # num_cols = data.select_dtypes(include=["number"]).columns.tolist()

        logging.info("Imputing NAN value from numric columns")

        if imputer is None:
            imputer = SimpleImputer(missing_values=np.nan, strategy="median")
            imputer.fit(data[num_cols])

        data_imputed_numeric = pd.DataFrame(
            imputer.transform(data[num_cols]),
            index=data.index,
            columns=num_cols
        )

        
        logging.info("Imputing NAN value from categorical columns")
        data = data.replace('?', np.nan)


        cat_cols = [col for col in data.columns if data[col].dtype == "object"]

        # cat_cols = data.select_dtypes(include=["object"]).columns.tolist()

        if imputer is None:
            imputer = SimpleImputer(missing_values=np.nan, strategy="constant" , fill_value="UNKNOWN")

            imputer.fit(data[cat_cols])

        data_imputed_categorical = pd.DataFrame(
            imputer.transform(data[cat_cols]),
            index=data.index,
            columns=cat_cols
        )



        data = pd.concat(
        [data_imputed_numeric, data_imputed_categorical],
        axis=1
        )

        return data
    '''



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




    '''

    def scaling_data(self , data, scaler=None):

        logging.info("----------------- Scaling Numeric value -------------------")
        num_cols = data.select_dtypes(include=['int' , 'int64', 'int32', 'float64']).columns.tolist()


        if len(num_cols) == 0:
            raise ValueError("No numeric columns found for scaling")

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(data[num_cols])

        data_scaled = pd.DataFrame(
            scaler.transform(data[num_cols]),
            index=data[num_cols].index,
            columns=num_cols
        )

        data = pd.concat(
            [data.drop(columns = data_scaled.columns) , data_scaled],
            axis = 1
        )

        return data
    '''


    '''
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
        

    '''

    def initiate_data_transformation(self) -> DataTransformationArtifact:

        """
        Initiates the data transformation component for the pipeline.
        """

        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            


            # Load train and test data

            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-test data loaded")


            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")


            # Apply Customer transformation 

            # Train Data

            input_feature_train_df = self._drop_id_columns(input_feature_train_df)
            input_feature_train_df = self.imputer_Num_cat(input_feature_train_df)

            '''
            input_feature_train_df = self.OHE_cat(input_feature_train_df)
            input_feature_train_df = self.OE_cat(input_feature_train_df)
            input_feature_train_df = self.label_encoding(input_feature_train_df)
            '''
            target_feature_train_df = train_df[TARGET_COLUMN].map({"N": 0, "Y": 1})



            # Test Data
            input_feature_test_df = self._drop_id_columns(input_feature_test_df)
            input_feature_test_df = self.imputer_Num_cat(input_feature_test_df)
            
            '''
            input_feature_test_df = self.OHE_cat(input_feature_test_df)
            input_feature_test_df = self.OE_cat(input_feature_test_df)
            input_feature_test_df = self.label_encoding(input_feature_test_df)
            '''
            target_feature_test_df = test_df[TARGET_COLUMN].map({"N": 0, "Y": 1})


            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Transformation done end to end to train-test df.")


            logging.info("Applying SMOTEENN for handling imbalanced dataset.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")

            smt = SMOTEENN(sampling_strategy="minority")
            
            # Train Resampling
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
           
            # but if you want to, use new variables to avoid confusion)
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )


            logging.info(f"Type of X_final: {type(input_feature_train_final)}")
            logging.info(f"Type of y_final: {type(target_feature_train_final)}")
            
            logging.info(f"Train Shapes: X={input_feature_train_final.shape}, y={target_feature_train_final.shape}")
            logging.info(f"Test Shapes: X={input_feature_test_final.shape}, y={target_feature_test_final.shape}")

          


            logging.info("Converting Sparse Matrix to Dense for concatenation...")
            
            
            X_train_dense = input_feature_train_final.toarray()
            X_test_dense = input_feature_test_final.toarray()

            
            y_train_2d = np.array(target_feature_train_final).reshape(-1, 1)
            y_test_2d = np.array(target_feature_test_final).reshape(-1, 1)

            
            train_arr = np.c_[X_train_dense, y_train_2d]
            test_arr = np.c_[X_test_dense, y_test_2d]

            logging.info(f"Final Concatenation Success! Train Shape: {train_arr.shape}")







            '''

            train_arr = np.c_[
                input_feature_train_final, 
                np.array(target_feature_train_final).reshape(-1, 1)
            ]
            
            test_arr = np.c_[
                input_feature_test_final, 
                np.array(target_feature_test_final).reshape(-1, 1)
            ]

            '''
            logging.info("feature-target concatenation done for train-test df.")

            logging.info(f"Train array Shape: {train_arr.shape}")
            logging.info(f"Test array Shape: {test_arr.shape}")

            save_object(self.data_transformation_config.transformed_object_file_path , preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path , array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path , array=test_arr)

            logging.info("Saving transformation object and transformed files.") 

            logging.info("Data transformation completed successfully")


            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        
            '''
            '''

        except Exception as e:
            raise MyException(e,sys)


