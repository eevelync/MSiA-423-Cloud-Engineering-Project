import boto3
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


s3 = boto3.client('s3')
obj2 = s3.get_object(Bucket = 'cloudengdfs', Key = "feature_eng_data.csv")
## grab the body of the object to get core data frame
df1 = pd.read_csv(obj2['Body'], index_col = None)


X = df1.drop(['Sale Price', "Buyer Region"], axis=1)
y = df1['Sale Price']
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

## Linear Model
base_lin_model =LinearRegression()
base_lin_model.fit(X_train, y_train)
y_pred = base_lin_model.predict(X_val)
mse_linear = round(mean_squared_error(y_val, y_pred), 2)
accuracy_linear = base_lin_model.score(X_train, y_train)
residuals_linear = np.sign(y_val - y_pred).mean()*100

## Xgboost Model

base_gb_model = GradientBoostingRegressor().fit(X_train, y_train)
y_pred = base_gb_model.predict(X_val)
accuracy_xgb = base_gb_model.score(X_train, y_train)
mse_xgb = mean_squared_error(y_val, y_pred)
residuals_xgb = np.sign(y_val - y_pred).mean()*100


with open('base_gb_model.pkl', 'wb') as f:
    pickle.dump(base_gb_model, f)

with open('base_lin_model.pkl', 'wb') as f:
    pickle.dump(base_lin_model, f)

s3 = boto3.client('s3')

#upload gb_boost to s3
bucket_name = 'cloudengdfs'
folder_name = 'model_objects/'
file_name = 'base_gb_model.pkl'
file_path = 'base_gb_model.pkl'
object_name = folder_name + file_name
s3.upload_file(file_path, bucket_name, object_name)

#upload linear model to s3
bucket_name = 'cloudengdfs'
folder_name = 'model_objects/'
file_name = 'base_lin_model.pkl'
file_path = 'base_lin_model.pkl'
object_name = folder_name + file_name
s3.upload_file(file_path, bucket_name, object_name)

