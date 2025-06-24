from churn_pred.config.configuration import ConfigurationManager
from churn_pred.components.c_03_data_transformation import DataTransformation
import pandas as pd
import pickle


class PredictionPipeline:
    def __init__(self):
        configuration_manager = ConfigurationManager()
        self.evaluation_config = configuration_manager.get_model_evaluation_config()
        self.transformation_config = configuration_manager.get_data_transformation_config()
    
    def data_preparation(self, data):
        data_transformation = DataTransformation(self.transformation_config)
        
        data_df = pd.read_csv(data)
        data_df = data_transformation.handling_missing_values(data_df)
        data_df = data_transformation.handling_datetime_features(data_df)
        data_df = data_transformation.feature_engineering(data_df)
        data_df = data_transformation.handling_categorical_features(data_df)
        
        prediction_df = data_df.drop(columns=['churn_next_month', 'early_churn'], errors='ignore')
        
        return prediction_df
    
    def loading_models(self):
        with open(self.evaluation_config.model_1, 'rb') as file:
            lg_model = pickle.load(file)
        
        with open(self.evaluation_config.model_1_scaler, 'rb') as file:
            lg_scaler = pickle.load(file)
        
        with open(self.evaluation_config.model_2, 'rb') as file:
            xgb_model = pickle.load(file)
        
        return lg_model, lg_scaler, xgb_model
    
    def predict(self, data):
        pred_df = self.data_preparation(data)
        
        lg_model, lg_scaler, xgb_model = self.loading_models()
        
        pred_df_scaled = lg_scaler.transform(pred_df)
        
        lg_pred_proba = lg_model.predict_proba(pred_df_scaled)[:,1]
        
        xgb_pred_proba = xgb_model.predict_proba(pred_df)[:,1]
        
        return lg_pred_proba, xgb_pred_proba
