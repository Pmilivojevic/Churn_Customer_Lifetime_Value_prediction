from churn_pred.entity.config_entity import ModelTrainerConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import pickle

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def data_prepatarion(self):
        train_data = pd.read_csv(self.config.train_file)
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        
        lg_y_train = train_data['churn_next_month']
        xgb_y_train = train_data['early_churn']
        
        X_train = train_data.drop(columns=['churn_next_month', 'early_churn'], errors='ignore')
        
        return X_train, lg_y_train, xgb_y_train
    
    def logistic_regression_trainer(self, X_train, y_train):
        params = self.config.model_params.logistic_regression.to_dict()
        model = LogisticRegression(**params)
        scaler = StandardScaler()
        
        pipe = Pipeline([
            ('scaler', scaler),
            ('clf', model)
        ])
        
        pipe.fit(X_train, y_train)
        
        with open(self.config.model_1,'wb') as f:
            pickle.dump(model, f)
        
        with open(self.config.model_1_scaler, 'wb') as f:
            pickle.dump(scaler, f)
    
    def xgb_classifier_trainer(self, X_train, y_train):
        params = self.config.model_params.xgb_classifier.to_dict()
        model = XGBClassifier()
        model.set_params(**params)
        
        model.fit(X_train, y_train)
        
        with open(self.config.model_2,'wb') as f:
            pickle.dump(model, f)
    
    def train(self):
        X_train, lg_y_train, xgb_y_train = self.data_prepatarion()
        
        self.logistic_regression_trainer(X_train, lg_y_train)
        
        self.xgb_classifier_trainer(X_train, xgb_y_train)
