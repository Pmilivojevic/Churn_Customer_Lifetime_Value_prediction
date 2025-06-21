from churn_pred.config.configuration import ConfigurationManager
from churn_pred.components.c_02_data_validation import DataValidation

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_dataset()
