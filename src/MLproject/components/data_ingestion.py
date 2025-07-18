import os
import urllib.request as request
import zipfile
from MLproject import logger
from MLproject.utils.common import get_size
from pathlib import Path
from MLproject.entity.config_entity import (DataIngestionConfig)
import pandas as pd


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")



    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted zip file to {unzip_path}")

        # Check for the CSV file and parse Date column
        csv_path = os.path.join(unzip_path, 'Dairy_Supply_Demand_20000.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            logger.info(f"CSV file loaded successfully with shape: {df.shape}")
        else:
            logger.error(f"Expected CSV file not found at {csv_path}")
  