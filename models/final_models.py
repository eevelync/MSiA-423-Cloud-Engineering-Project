import boto3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
s3 = boto3.client('s3')
# read in data
# Model is trained on original cleaned data that has been cleaned and transformed
obj2 = s3.get_object(Bucket = 'cloudengdfs', Key = "feature_eng_data.csv")
## grab the body of the object to get core data frame
df1 = pd.read_csv(obj2['Body'], index_col = None)


X = df1.drop(['Sale Price', "Buyer Region"], axis=1)
y = df1['Sale Price']
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Download gb_model from S3
bucket_name = 'cloudengdfs'
gb_object_name = 'model_objects/base_gb_model.pkl'
gb_file_path = 'base_gb_model.pkl'
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

gb_grid_search = GridSearchCV(base_gb_model, gb_param_grid, cv=5)
gb_grid_search.fit(X_train, y_train)

best_gb_model = gb_grid_search.best_estimator_
best_gb_params = gb_grid_search.best_params_


with open('best_gb_model.pkl', 'wb') as f:
    pickle.dump(best_gb_model, f)

# Upload the best_gb_model file to S3
bucket_name = 'cloudengdfs'
object_name = 'model_objects/best_gb_model.pkl'
file_path = 'best_gb_model.pkl'

s3.upload_file(file_path, bucket_name, object_name)
