import boto3
import pandas as pd
import json
import requests
import time
from datetime import date
from datetime import datetime
from typing import List, Union


def lambda_handler(event, context):
    s3 = boto3.client('s3')
    
    obj = s3.get_object(Bucket = 'cloudengdfs', Key = 'StockX-Data-Contest-2019-3.csv')
    df = pd.read_csv(obj['Body'], index_col = None)
    models = df['Sneaker Name'].unique()
    
    def search(query: str, item: str) -> str:
        
        """
        Search for an item in StockX API and return a specific market value.

        Args:
            query (str): The search query to be used.
            item (str): The specific item to retrieve from the API response.

        Returns:
            str: The market value of the requested item.

        Raises:
            IndexError: If the API response does not contain the requested item.
            requests.exceptions.RequestException: If there is an error in making the API request.
         """
        # Define the headers to use for the request
        headers = {
            'accept': 'application/json',
            'accept-encoding': 'utf-8',
            'accept-language': 'en-GB,en;q=0.9',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
            'x-requested-with': 'XMLHttpRequest',
            'app-platform': 'Iron',
            'app-version': '2022.05.08.04',
            'referer': 'https://stockx.com/'
        }
        
        # Send the request to the API and parse the response as JSON
        url = f'https://stockx.com/api/browse?_search={query}'
        html = requests.get(url=url, headers=headers)
        output = json.loads(html.text)
    
        # Return the requested item from the API response
        return output['Products'][0]['market'][item]
    
    def brand(query: str) -> str:
        """
        Get the brand of a shoe from the StockX API.

        Args:
            query (str): The search query to be used.

        Returns:
            str: The brand of the shoe.

        Raises:
            IndexError: If the API response does not contain the brand information.
            requests.exceptions.RequestException: If there is an error in making the API request.
        """
        # Define the headers to use for the request
        headers = {
            'accept': 'application/json',
            'accept-encoding': 'utf-8',
            'accept-language': 'en-GB,en;q=0.9',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
            'x-requested-with': 'XMLHttpRequest',
            'app-platform': 'Iron',
            'app-version': '2022.05.08.04',
            'referer': 'https://stockx.com/'
        }
        
        # Send the request to the API and parse the response as JSON
        url = f'https://stockx.com/api/browse?_search={query}'
        html = requests.get(url=url, headers=headers)
        output = json.loads(html.text)
    
        # Extract the brand of the shoe from the product information
        brand = output['Products'][0]['brand']
        return brand
    
    def dataframe(model: str) -> List[Union[float, str]]:
        """
        Create a dataframe for a given model by retrieving specific information from the StockX API.

        Args:
            model (str): The model of the shoe to create the dataframe for.

        Returns:
            List[Union[float, str]]: The dataframe containing the last sale value and size for the given model.

        Raises:
            requests.exceptions.RequestException: If there is an error in making the API request.
        """
        em_list = []
        for i in models:
            if i == model:
                em_list.append(float(search(model, "lastSale")))
                em_list.append((search(model, "lastSaleSize")))
        return em_list
        
    def f_df(model: str) -> pd.DataFrame:
        """
        Create a formatted Pandas DataFrame for a given model by retrieving specific information from the StockX API.

        Args:
            model (str): The model of the shoe to create the dataframe for.

        Returns:
            pd.DataFrame: The formatted dataframe containing information for the given model.

        Raises:
            requests.exceptions.RequestException: If there is an error in making the API request.
        """
        # Retrieve data for the given model and format it as a Pandas DataFrame
        vals = dataframe(model)
        temp_df = df[df["Sneaker Name"] == model]
        order_date = search(model, "lastSaleDate")
        order_date = datetime.strptime(order_date, '%Y-%m-%dT%H:%M:%SZ')
        order_date = order_date.date()
        p_df = {"Order Date": order_date,
                "Brand": brand(model),
                "Sneaker Name": model,
                "Sale Price": vals[0],
                "Retail Price": temp_df["Retail Price"].unique(),
                "Release Date": temp_df['Release Date'].unique(),
                "Shoe Size": vals[1],
                "Buyer Region": "Illinois"}
        p_df = pd.DataFrame(p_df, index=[0])
        p_df['Shoe Size'] = p_df['Shoe Size'].str.replace('W', '')
        p_df['Shoe Size'] = p_df['Shoe Size'].str.replace('K', '')
        p_df['Shoe Size'] = float(p_df['Shoe Size'])
        return p_df
        
    dfs = []
    for i in models:
        tr3 = f_df(i)
        dfs.append(tr3)
        time.sleep(5)
    dfs = pd.concat(dfs, ignore_index = True)

    bucket_name = 'cloudengdfs'
    key_name = 'stockx_data.csv'
    
    final_df = dfs.to_csv(index=False).encode('utf-8')
    s3.put_object(Bucket=bucket_name, Key=key_name, Body=final_df)

    return {
        'statusCode': 200,
        'body': f'Data for new sales successfully retrieved and uploaded to S3'
    }

