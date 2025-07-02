import pandas as pd
import os
from MLproject import logger
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
import joblib
from MLproject.entity.config_entity import ModelTrainerConfig
from statsmodels.tsa.arima.model import ARIMA



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop(self.config.target_columns, axis=1)
        test_x = test_data.drop(self.config.target_columns, axis=1)
        train_y = train_data[self.config.target_columns]
        test_y = test_data[self.config.target_columns]


        base_model = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(train_x, train_y)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))

    def train_arima(self, order=(2,1,2)):
        """
        Train an ARIMA model for each target column (univariate time series forecasting).
        Saves each ARIMA model as a separate file in the root_dir.
        """
        train_data = pd.read_csv(self.config.train_data_path, parse_dates=['Date'])
        train_data = train_data.sort_values('Date')
        arima_models = {}
        for target_col in self.config.target_columns:
            print(f"Training ARIMA for {target_col}...")
            series = train_data[target_col]
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            # Save the fitted model
            model_path = os.path.join(self.config.root_dir, f"arima_{target_col}.pkl")
            joblib.dump(model_fit, model_path)
            arima_models[target_col] = model_path
            logger.info(f"Saved ARIMA model for {target_col} at {model_path}")
        return arima_models

