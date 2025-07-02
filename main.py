from MLproject import logger
from MLproject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from MLproject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from MLproject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from MLproject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from MLproject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
import pandas as pd
import joblib
import os

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataValidationTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataTransformationTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelTrainerTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model evaluation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelEvaluationTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# --- Export predictions for business review ---
try:
    # Load test data and model
    test_path = "artifacts/model_evaluation/test.csv"
    model_path = "artifacts/model_trainer/model.joblib"
    if os.path.exists(test_path) and os.path.exists(model_path):
        test_df = pd.read_csv(test_path)
        model = joblib.load(model_path)
        # Assume target columns are known (update as needed)
        target_columns = [
            'Milk_500ml_Demand', 'Milk_1L_Demand', 'Butter_Demand', 'Cheese_Demand', 'Yogurt_Demand'
        ]
        test_x = test_df.drop(target_columns, axis=1)
        preds = model.predict(test_x)
        preds_df = pd.DataFrame(preds, columns=[f'Predicted_{col}' for col in target_columns])
        results = pd.concat([test_df.reset_index(drop=True), preds_df], axis=1)
        results.to_csv("artifacts/model_evaluation/predictions_for_business.csv", index=False)
        logger.info("Exported predictions for business review to artifacts/model_evaluation/predictions_for_business.csv")
except Exception as e:
    logger.exception(e)

