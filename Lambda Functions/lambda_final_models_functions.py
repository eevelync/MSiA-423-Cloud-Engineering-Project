import boto3
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


# Configure logging
logging.basicConfig(level=logging.INFO)

def lambda_handler(event: dict, context: object) -> dict:
    """
    Lambda function handler that performs model tuning on a pre-trained gradient boosting model,
    saves the best model as a pickle file, and uploads it back to S3.

    Args:
        event (dict): The event data passed to the Lambda function.
        context (object): The runtime information of the Lambda function.

    Returns:
        dict: A dictionary containing the response status code and a message.
    """

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Read data from S3
    obj2 = s3.get_object(Bucket='cloudengdfs', Key='feature_eng_data.csv')
    df1 = pd.read_csv(obj2['Body'], index_col=None)

    X = df1.drop(['Sale Price', 'Buyer Region'], axis=1)
    y = df1['Sale Price']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Download gb_model from S3
    bucket_name = 'cloudengdfs'
    gb_object_name = 'model_objects/base_gb_model.pkl'
    gb_file_path = '/tmp/base_gb_model.pkl'
    s3.download_file(bucket_name, gb_object_name, gb_file_path)

    # Deserialize gb_model
    with open(gb_file_path, 'rb') as f:
        base_gb_model = pickle.load(f)

    # Model tuning
    gb_param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7]
    }

    logging.info("Performing model tuning...")
    gb_grid_search = GridSearchCV(base_gb_model, gb_param_grid, cv=5)
    gb_grid_search.fit(X_train, y_train)

    best_gb_model = gb_grid_search.best_estimator_
    best_gb_params = gb_grid_search.best_params_

    # Save the best_gb_model as a pickle file
    logging.info("Saving best model as pickle file...")
    with open('/tmp/best_gb_model.pkl', 'wb') as f:
        pickle.dump(best_gb_model, f)

    # Upload the best_gb_model file to S3
    bucket_name = 'cloudengdfs'
    object_name = 'model_objects/best_gb_model.pkl'
    file_path = '/tmp/best_gb_model.pkl'
    s3.upload_file(file_path, bucket_name, object_name)

    logging.info("Model objects uploaded to S3.")

    return {
        'statusCode': 200,
        'body': 'Model Objects uploaded to S3'
    }
