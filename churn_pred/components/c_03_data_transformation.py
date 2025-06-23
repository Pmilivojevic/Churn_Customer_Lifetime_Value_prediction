import os
import pandas as pd
import numpy as np
# from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from churn_pred.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def loading_data(self):
        df = pd.read_excel(self.config.data_file, engine='openpyxl')
        
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
        
        financial_features = df.select_dtypes(include=np.float64).columns.to_list()
        
        print("Names of financial features:", financial_features)
        
        for col in financial_features:
            df[f'log_{col}'] = np.log1p(df[col])
        
        return df
        
    def feature_engineering(self, df):
        df = df.sort_values('activity_month')
        
        df['months_active'] = ((
            df['activity_month'].dt.year - df['ftd_date'].dt.year
        ) * 12 + (
            df['activity_month'].dt.month - df['ftd_date'].dt.month
        )).astype(int)
        
        # Find last month of activity for each player
        last_months = df.groupby('account_id')['activity_month'].max().reset_index()
        last_months.columns = ['account_id', 'last_activity']
        
        # Merge last_activity back to main df
        df = df.merge(last_months, on='account_id')
        
        # Calculate months_since_last_activity
        df['months_since_last_activity'] = (
            (
                df['last_activity'].dt.to_period('M') - df['activity_month'].dt.to_period('M')
            ).apply(lambda x: x.n)
        )
        
        df['churned'] = (
            df['months_since_last_activity'] >= self.config.transfrmation_params.churn_months_thr
        ).astype(int)
        
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
        
        return df, encoded.columns.to_list()
    
    def train_test_split_and_save(self, df, encoded_columns):
        feature_columns = ['months_active', 'log_total_deposit', 'log_total_handle', 'log_total_ngr'] + encoded_columns
        data_df = df[feature_columns]
        
        split_idx = int((1 - self.config.transfrmation_params.test_size) * len(data_df))
        
        train_df = data_df.iloc[:split_idx]
        test_df = data_df.iloc[split_idx:]
        
        train_df.to_csv(self.config.train_file, index=False)
        test_df.to_csv(self.config.test_file, index=False)
    
    def transformation_compose(self):
        if self.config.dataset_val_status:
            if not os.path.exists(self.config.train_file) and not os.path.exists(self.config.test_file):
                df = self.loading_data()
                df = self.handling_missing_values(df)
                df = self.handling_datetime_features(df)
                df = self.log_transform(df)
                df = self.feature_engineering(df)
                df, encoded_columns = self.handling_categorical_features(df)
                self.train_test_split_and_save(df, encoded_columns)
            else:
                print("The dataset has already been split and prepared.")
        else:
            print("Dataset is not valid!")
