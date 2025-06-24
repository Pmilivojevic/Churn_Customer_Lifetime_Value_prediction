import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from churn_pred.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def loading_data(self):
        df = pd.read_excel(self.config.data_file)
        
        return df
    
    def handling_missing_values(self, df):
        print("Features names:",df.columns.to_list())
        
        null = df.isnull().sum()
        null = null[null > 0]
        print("Names of features of missing values:",null.index.to_list())
        
        df['ftd_date'] = df['ftd_date'].fillna(df['reg_date'])
        df['qp_date'] = df['qp_date'].fillna(df['ftd_date'])
        df['total_handle'] = df['total_handle'].fillna(0.0)

        print(df.isnull().sum())
        
        return df
    
    def handling_datetime_features(self, df):
        print("Features names:",df.columns.to_list())
        
        datetime_features = df.select_dtypes(include='datetime').columns.to_list()
        
        print("Names of datetime features:", datetime_features)
        
        df[datetime_features] = df[datetime_features].apply(pd.to_datetime)
        
        return df
    
    def log_transform(self, df):
        print("Features names:",df.columns.to_list())
        
        financial_features = ['total_deposit', 'total_handle', 'total_ngr']
        
        print("Names of financial features:", financial_features)
        
        for col in financial_features:
            df[f'log_{col}'] = np.log1p(df[col])
        
        df = df.drop(columns=financial_features)
        
        return df
        
    def feature_engineering(self, df):
        # df = df.sort_values('activity_month')
        
        df = df.sort_values(['account_id', 'activity_month'])
        
        df['months_active'] = ((
            df['activity_month'].dt.year - df['ftd_date'].dt.year
        ) * 12 + (
            df['activity_month'].dt.month - df['ftd_date'].dt.month
        )).astype(int)
        
        df['has_qp'] = df['qp_date'].notnull().astype(int)
        df['days_ftd_to_qp'] = (df['qp_date'] - df['ftd_date']).dt.days.fillna(-1)
        df['reg_date'] = df['reg_date'].dt.month

        df['next_activity_month'] = df.groupby('account_id')['activity_month'].shift(-1)
        df['months_to_next_activity'] = (df['next_activity_month'] - df['activity_month']).dt.days / 30

        # Define churn: no activity in next 2 months = churn
        df['churn_next_month'] = (
            df['months_to_next_activity'] >= self.config.transfrmation_params.no_activity_thr
        ).astype(int)
        
        # Find player's last activity month
        last_month_df = df.groupby('account_id')['activity_month'].max().reset_index()
        last_month_df.rename(columns={'activity_month': 'last_activity_month'}, inplace=True)

        # Merge back
        df = df.merge(last_month_df, on='account_id')

        # Compute months since last activity per record
        df['months_since_last_activity'] = ((df['last_activity_month'] - df['activity_month']).dt.days) / 30

        # Label churners: no activity ≥ 2 months after last activity
        df['churned'] = (
            df['months_since_last_activity'] >= self.config.transfrmation_params.churn_months_thr
        ).astype(int)
        
        # 
        df['early_churn'] = (
            df['months_active'] <= self.config.transfrmation_params.early_churn_thr
        ).astype(int)
        
        df = df.drop(columns=[
            'activity_month',
            'next_activity_month',
            'months_to_next_activity',
            'last_activity_month',
            'months_since_last_activity',
            'ftd_date',
            'churned',
            'account_id',
            'tracker_id',
            'qp_date'
        ], errors='ignore')
        
        return df
    
    def handling_categorical_features(self, df):
        print("Features names:",df.columns.to_list())
        
        categorical_features = df.select_dtypes(include='object').columns.to_list()
        
        print("Names of categorical features:", categorical_features)
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = pd.DataFrame(
            encoder.fit_transform(df[categorical_features]),
            columns=encoder.get_feature_names_out(categorical_features),
            index=df.index
        )
        
        df = pd.concat([df, encoded], axis=1)
        df.drop(categorical_features, axis=1, inplace=True)
        
        return df
    
    def train_test_split_and_save(self, df):
        split_idx = int((1 - self.config.transfrmation_params.test_size) * len(df))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        train_df.to_csv(self.config.train_file, index=False)
        test_df.to_csv(self.config.test_file, index=False)
    
    def transformation_compose(self):
        if self.config.dataset_val_status:
            if not os.path.exists(self.config.train_file) and not os.path.exists(self.config.test_file):
                df = self.loading_data()
                df = self.handling_missing_values(df)
                df = self.handling_datetime_features(df)
                # df = self.log_transform(df)
                df = self.feature_engineering(df)
                df = self.handling_categorical_features(df)
                self.train_test_split_and_save(df)
            else:
                print("The dataset has already been split and prepared.")
        else:
            print("Dataset is not valid!")
