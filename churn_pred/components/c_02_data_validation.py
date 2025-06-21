import pandas as pd
from churn_pred import logger
from churn_pred.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_dataset(self) -> bool:
        validation_status = True
        
        try:
            # Read dataset
            data_df = pd.read_excel(self.config.local_data_file, engine='openpyxl')
            
            # Validate column names
            if set(data_df.columns) != set(self.config.all_schema):
                validation_status = False
                logger.info("Columns in the dataset CSV file do not match the schema!")
            
            # Write final validation status once
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status

        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            raise
