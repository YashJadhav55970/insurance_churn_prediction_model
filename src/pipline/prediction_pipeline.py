import sys
from src.entity.config_entity import InsuranceConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class InsuranceData:
    def __init__(self,
                 incident_severity,
                 insured_hobbies,
                 incident_type,
                 collision_type,
                 incident_state,
                 authorities_contacted,
                 property_claim,
                 vehicle_claim,
                 policy_annual_premium,
                 insured_zip,
                 policy_number,
                 total_claim_amount,
                 months_as_customer,
                 injury_claim,
                 age,
                 insured_occupation,
                 capital_gains,
                 fraud_reported
                 ):
        
        try:
            self.incident_severity = incident_severity
            self.insured_hobbies = insured_hobbies
            self.incident_type = incident_type
            self.collision_type = collision_type
            self.incident_state = incident_state
            self.authorities_contacted = authorities_contacted
            self.property_claim = property_claim
            self.vehicle_claim = vehicle_claim
            self.policy_annual_premium = policy_annual_premium
            self.insured_zip = insured_zip
            self.policy_number = policy_number
            self.total_claim_amount = total_claim_amount
            self.months_as_customer = months_as_customer
            self.injury_claim = injury_claim
            self.age = age
            self.insured_occupation = insured_occupation
            self.capital_gains = capital_gains
            self.fraud_reported = fraud_reported
        except Exception as e:
            raise MyException(e, sys) from e
        
    def get_insurance_data_as_dict(self):
        """
        This function returns a dictionary from InsuranceData class input
        """
        logging.info("Entered get_insurance_data_as_dict method as InsuranceData class")
        
        try:
            input_data = {
                "incident_severity": [self.incident_severity],
                "insured_hobbies": [self.insured_hobbies],
                "incident_type": [self.incident_type],
                "collision_type": [self.collision_type],
                "incident_state": [self.incident_state],
                "authorities_contacted": [self.authorities_contacted],
                "property_claim": [self.property_claim],
                "vehicle_claim": [self.vehicle_claim],
                "policy_annual_premium": [self.policy_annual_premium],
                "insured_zip": [self.insured_zip],
                "policy_number": [self.policy_number],
                "total_claim_amount": [self.total_claim_amount],
                "months_as_customer": [self.months_as_customer],
                "injury_claim": [self.injury_claim],
                "age": [self.age],
                "insured_occupation": [self.insured_occupation],
                "capital-gains": [self.capital_gains],
                "fraud_reported": [self.fraud_reported],
            }
            return input_data
        except Exception as e:
            raise MyException(e, sys) from e
        
    # Data in dataframe format
    def get_insurance_input_data_frame(self) -> DataFrame:
        try:
            insurance_input_dict = self.get_insurance_data_as_dict()
            return DataFrame(insurance_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e
        

# InsuranceData Classifier class
class InsuranceDataClassifier:
    def __init__(self, prediction_pipeline_config: InsuranceConfig = InsuranceConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        
        except Exception as e:
            raise MyException(e, sys)
        
    def predict(self, dataframe) -> str:
        try:
            logging.info("Entered predict method of InsuranceDataClassifier class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path
            )
            result = model.predict(dataframe)
            return result
        
        except Exception as e:
            raise MyException(e, sys)