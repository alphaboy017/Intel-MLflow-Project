import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from MLproject.entity.config_entity import ModelEvaluationConfig
from MLproject.utils.common import save_json
from pathlib import Path
import matplotlib.pyplot as plt


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100  # avoid division by zero
        return rmse, mae, r2, mape
    


    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop(self.config.target_columns, axis=1)
        test_y = test_data[self.config.target_columns]
        preds = model.predict(test_x)
        if len(self.config.target_columns) == 1:
            preds = preds.reshape(-1, 1)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Dairy_Demand_Forecasting")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        scores = {}
        with mlflow.start_run(run_name="ElasticNet_MultiTarget"):
            for idx, col in enumerate(self.config.target_columns):
                y_true = test_y.iloc[:, idx]
                y_pred = preds[:, idx]
                rmse, mae, r2, mape = self.eval_metrics(y_true, y_pred)
                scores[col] = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
                mlflow.log_metric(f"rmse_{col}", rmse)
                mlflow.log_metric(f"mae_{col}", mae)
                mlflow.log_metric(f"r2_{col}", r2)
                mlflow.log_metric(f"mape_{col}", mape)

                # Save actual vs predicted plot
                plt.figure(figsize=(10, 4))
                plt.plot(y_true.values, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'Actual vs Predicted for {col}')
                plt.xlabel('Sample')
                plt.ylabel(col)
                plt.legend()
                plot_path = os.path.join(self.config.root_dir, f'actual_vs_pred_{col}.png')
                plt.savefig(plot_path)
                plt.close()
                mlflow.log_artifact(plot_path)

            save_json(path=Path(self.config.metric_file_name), data=scores)
            mlflow.log_params(self.config.all_params)
            # Log dairy-specific parameters
            mlflow.log_param("lag_features", "lag1, lag7")
            mlflow.log_param("rolling_window", 7)
            mlflow.log_param("feature_engineering", "date features, lags, rolling means")
            # Log schema.yaml as an artifact
            schema_path = os.path.abspath("schema.yaml")
            if os.path.exists(schema_path):
                mlflow.log_artifact(schema_path)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")

    
