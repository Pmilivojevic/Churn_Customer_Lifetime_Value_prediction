from churn_pred.config.configuration import ConfigurationManager
from churn_pred.components.c_03_data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transformation_compose()
