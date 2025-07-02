import os
from MLproject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from MLproject.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config  
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        # Load the data and parse the Date column
        data = pd.read_csv(self.config.data_path, parse_dates=['Date'])

        # 1. Extract date features for seasonality analysis
        data['day_of_week'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        data['month'] = data['Date'].dt.month
        data['year'] = data['Date'].dt.year

        # 2. Create lag features to capture temporal dependencies
        # Lag 1: previous day's value, Lag 7: value from previous week
        for col in [
            'Milk_500ml_Demand', 'Milk_1L_Demand', 'Butter_Demand', 'Cheese_Demand', 'Yogurt_Demand'
        ]:
            data[f'{col}_lag1'] = data[col].shift(1)
            data[f'{col}_lag7'] = data[col].shift(7)

        # 3. Calculate rolling averages (7-day mean) to smooth out short-term fluctuations
        for col in [
            'Milk_500ml_Demand', 'Milk_1L_Demand', 'Butter_Demand', 'Cheese_Demand', 'Yogurt_Demand'
        ]:
            data[f'{col}_roll7'] = data[col].rolling(window=7).mean()

        # Drop rows with NaN values created by shifting/rolling (first 7 days will have NaNs)
        data = data.dropna().reset_index(drop=True)

        # 4. Sort by date to ensure time order
        data = data.sort_values('Date').reset_index(drop=True)
        split_idx = int(len(data) * 0.75)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]

        # 5. Save the split datasets to CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets (time-ordered)")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        