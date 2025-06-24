from churn_pred.entity.config_entity import ModelEvaluationConfig
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from churn_pred.utils.main_utils import save_json, plot_confusion_matrix

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def data_prepatarion(self):
        test_data = pd.read_csv(self.config.test_file)
        test_data = test_data.replace([np.inf, -np.inf], np.nan)
        
        lg_y_test = test_data['churn_next_month']
        xgb_y_test = test_data['early_churn']
        
        X_test = test_data.drop(columns=['churn_next_month', 'early_churn'])
        
        return X_test, lg_y_test, xgb_y_test
    
    def evaluate_lg_model(self, X_test, y_test):
        with open(self.config.model_1, 'rb') as file:
            model = pickle.load(file)
        
        with open(self.config.model_1_scaler, 'rb') as file:
            scaler = pickle.load(file)
        
        X_test = scaler.transform(X_test)
        
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)

        # Evaluate
        auc = {}
        auc['auc'] = roc_auc_score(y_test, y_pred_proba)
        save_json(self.config.model_1_stats, auc, 'logistic_regression_auc')
        
        report = classification_report(y_test, y_pred, output_dict=True)
        save_json(self.config.model_1_stats, report, 'logistic_regression_report')
        
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, self.config.model_1_stats, 'logistic_regression')
        
        # print(auc, report)
    
    def evaluate_xgb_model(self, X_test, y_test):
        with open(self.config.model_2, 'rb') as file:
            model = pickle.load(file)
        
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)

        # Evaluate
        auc = {}
        auc['auc'] = roc_auc_score(y_test, y_pred_proba)
        save_json(self.config.model_2_stats, auc, 'xgb_classifier_auc')
        
        report = classification_report(y_test, y_pred, output_dict=True)
        save_json(self.config.model_2_stats, report, 'xgb_classifier_report')
        
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, self.config.model_2_stats, 'xgb_classifier')
        
        # print(report)
    
    def evaluation(self):
        X_test, lg_y_test, xgb_y_test = self.data_prepatarion()
        
        self.evaluate_lg_model(X_test, lg_y_test)
        
        self.evaluate_xgb_model(X_test, xgb_y_test)
