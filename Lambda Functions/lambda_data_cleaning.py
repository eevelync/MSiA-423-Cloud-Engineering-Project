import pandas as pd
import boto3
from datetime import date
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def lambda_handler(event: dict, context: object) -> dict:
    """
    Lambda function handler that retrieves data from S3, performs data cleaning and transformation,
    and uploads the cleaned data back to S3.

    Args:
        event (dict): The event data passed to the Lambda function.
        context (object): The runtime information of the Lambda function.

    Returns:
        dict: A dictionary containing the response status code and a message.
    """

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Retrieve the 'stockx_data.csv' file from S3
    obj = s3.get_object(Bucket='cloudengdfs', Key='stockx_data.csv')
    dfs = pd.read_csv(obj['Body'], index_col=None)

    # Retrieve the 'StockX-Data-Contest-2019-3.csv' file from S3
    obj2 = s3.get_object(Bucket='cloudengdfs', Key='StockX-Data-Contest-2019-3.csv')
    df = pd.read_csv(obj2['Body'], index_col=None)

    # Perform data cleaning and transformation
    df["Order Date"] = date.today()
    df["Sale Price"] = df["Sale Price"].str.replace("$", "").str.replace(",", "")
    df["Retail Price"] = df["Retail Price"].str.replace("$", "").str.replace(",", "")
    df[["Sale Price", "Retail Price"]] = df[["Sale Price", "Retail Price"]].astype(int)
    df['Brand'] = df['Brand'].str.replace(' Yeezy', 'adidas')
    df['Brand'] = df['Brand'].str.replace('Off-White', 'nike')

    dfs = dfs.rename(columns={"Sale Price": "Last Sale Price"})
    dfs['Retail Price'] = dfs['Retail Price'].str.replace("$", '').astype(int)
    dfs["Order Date"] = pd.to_datetime(dfs["Order Date"])
    dfs["Release Date"] = pd.to_datetime(dfs["Release Date"])
    dfs['Brand'] = dfs['Brand'].str.replace('Nike', 'nike')
    dfs['Brand'] = dfs['Brand'].replace('Jordan', 'nike')
    dfs = pd.get_dummies(dfs, columns=['Brand', 'Sneaker Name'])

    dfs['markup'] = (dfs['Last Sale Price'] - dfs['Retail Price']) / dfs['Retail Price']
    mean_resale_price_by_model = pd.DataFrame(df.groupby(['Sneaker Name'])['Sale Price'].mean()).reset_index()
    dfs['mean_resale_price_by_model'] = mean_resale_price_by_model['Sale Price']
    dfs['age_in_days'] = (dfs['Order Date'] - dfs['Release Date']).dt.days
    dfs.drop(['Order Date', 'Release Date'], axis=1, inplace=True)

    # Upload the cleaned data back to S3
    bucket_name = 'cloudengdfs'
    key_name = 'cleaned_data.csv'
    final_df = dfs.to_csv(index=False).encode('utf-8')
    s3.put_object(Bucket=bucket_name, Key=key_name, Body=final_df)

    logging.info("Cleaned data uploaded to S3.")

    return {
        'statusCode': 200,
        'body': 'Data for new sales successfully retrieved and uploaded to S3'
    }
