from churn_pred.constants import *
from churn_pred.utils.main_utils import create_directories, read_yaml
from churn_pred.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_file_path = CONFIG_FILE_PATH,
        params_file_path = PARAMS_FILE_PATH,
        schema_file_path = SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        transformation_params = self.params.data_transformation
        dataset_val_status_file = self.config.data_validation.STATUS_FILE
        
        with open(dataset_val_status_file, 'r') as f:
            status = f.read()
        
        status = bool(str.split(status)[-1])
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_file=Path(config.data_file),
            train_file=Path(config.train_file,),
            test_file=Path(config.test_file),
            transfrmation_params=transformation_params,
            dataset_val_status=status
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_params
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_file=Path(config.train_file),
            model_1=Path(config.model_1),
            model_1_scaler=Path(config.model_1_scaler),
            model_2=Path(config.model_2),
            model_params=params
        )
        
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([config.root_dir, config.model_1_stats, config.model_2_stats])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_file=Path(config.test_file),
            model_1=Path(config.model_1),
            model_1_scaler=Path(config.model_1_scaler),
            model_2=Path(config.model_2),
            model_1_stats=Path(config.model_1_stats),
            model_2_stats=Path(config.model_2_stats)
        )
        
        return model_evaluation_config
