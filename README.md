# MLflow-Project

## Project Transformation Summary

This project was originally a generic MLflow-based machine learning pipeline template. It has been transformed into a specialized demand forecasting solution for Flavi Dairy Solutions, Ahmedabad. The focus is now on predicting dairy supply and inventory demand to optimize processing, packaging, and storage capacities. The project is now ready for deployment on platforms like Render, with experiment tracking handled via MLflow and DagsHub.

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/alphaboy017/MLflow-Project
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up your local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/alphaboy017/MLflow-Project.mlflow \
MLFLOW_TRACKING_USERNAME=alphaboy017 \
MLFLOW_TRACKING_PASSWORD=2261233cd3fb0adfeeafe605203546596d34800a \
python script.py

Run this to export as env variables:
//Note that 'Export' command works in Linux ONLY
use this for Windows 
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/alphaboy017/MLflow-Project.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "alphaboy017"
$env:MLFLOW_TRACKING_PASSWORD = "your_dagshub_access_token_here" 
```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/alphaboy017/MLflow-Project.mlflow

export MLFLOW_TRACKING_USERNAME=alphaboy017 

export MLFLOW_TRACKING_PASSWORD=2261233cd3fb0adfeeafe605203546596d34800a

```



## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your experiments
 - Logging & tagging your model


## 🥛 Dairy Demand Forecasting

**Organisation:** Flavi Dairy Solutions, Ahmedabad  
**Category:** Industry defined problem

### Project Description
Maximizing the utilization of processing, packaging, and storage capacities is crucial for cost-efficiency and profitability in dairy plants. This project aims to forecast demand and optimize supply/inventory, addressing challenges such as fluctuating milk supply, demand variability, SKU changes, equipment downtime, and inefficient scheduling. Improved forecasting will help reduce underperformance, lower production costs, and enhance ROI for fixed assets.

### Objectives
- Predict future demand for dairy products
- Optimize inventory and supply chain operations
- Improve plant capacity utilization

### Success Metrics
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **Capacity Utilization Rate**
- **ROI Improvement**


