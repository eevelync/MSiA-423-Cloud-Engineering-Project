import boto3
import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def lambda_handler(event: dict, context: object) -> dict:
    """
    Lambda function handler that trains linear regression and gradient boosting models,
    saves them as pickle files, and uploads them to an S3 bucket.

    Args:
        event (dict): The event data passed to the Lambda function.
        context (object): The runtime information of the Lambda function.

    Returns:
        dict: A dictionary containing the response status code and a message.
    """

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Retrieve the 'feature_eng_data.csv' file from S3
    obj2 = s3.get_object(Bucket='cloudengdfs', Key='feature_eng_data.csv')

    # Read the CSV file into a pandas DataFrame
    df1 = pd.read_csv(obj2['Body'], index_col=None)

    # Prepare input features and target variable
    X = df1.drop(['Sale Price', 'Buyer Region'], axis=1)
    y = df1['Sale Price']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    logging.info("Training Linear Regression model...")
    base_lin_model = LinearRegression()
    base_lin_model.fit(X_train, y_train)
    y_pred = base_lin_model.predict(X_val)
    mse_linear = round(mean_squared_error(y_val, y_pred), 2)
    accuracy_linear = base_lin_model.score(X_train, y_train)
    residuals_linear = np.sign(y_val - y_pred).mean() * 100

    # Train a gradient boosting model
    logging.info("Training Gradient Boosting model...")
    base_gb_model = GradientBoostingRegressor().fit(X_train, y_train)
    y_pred = base_gb_model.predict(X_val)
    accuracy_xgb = base_gb_model.score(X_train, y_train)
    mse_xgb = mean_squared_error(y_val, y_pred)
    residuals_xgb = np.sign(y_val - y_pred).mean() * 100

    # Save the gradient boosting model as a pickle file
    logging.info("Saving Gradient Boosting model as pickle file...")
    with open('/tmp/base_gb_model.pkl', 'wb') as f:
        pickle.dump(base_gb_model, f)

    # Save the linear regression model as a pickle file
    logging.info("Saving Linear Regression model as pickle file...")
    with open('/tmp/base_lin_model.pkl', 'wb') as f:
        pickle.dump(base_lin_model, f)

    # Upload the gradient boosting model to S3
    logging.info("Uploading Gradient Boosting model to S3...")
    bucket_name = 'cloudengdfs'
    folder_name = 'model_objects/'
    file_name = 'base_gb_model.pkl'
    file_path = '/tmp/base_gb_model.pkl'
    object_name = folder_name + file_name
    s3.upload_file(file_path, bucket_name, object_name)

    # Upload the linear regression model to S3
    logging.info("Uploading Linear Regression model to S3...")
    file_name = 'base_lin_model.pkl'
    file_path = '/tmp/base_lin_model.pkl'
    object_name = folder_name + file_name
    s3.upload_file(file_path, bucket_name, object_name)

    logging.info("Model objects uploaded to S3.")

    return {
        'statusCode': 200,
        'body': 'Model Objects uploaded to S3'
    }
