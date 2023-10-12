import os
import boto3
import pickle
import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from typing import Tuple



@st.cache_data(ttl=1 * 24 * 60 * 60)
def dataframes() -> Tuple:
        
    """
    Retrieve dataframes from S3.

    Returns:
        tuple: A tuple of dataframes.
    """

    s3 = boto3.client('s3')
    # Read original_cleaned_data.csv
    obj = s3.get_object(Bucket='cloudengdfs', Key='original_cleaned_data.csv')
    df = pd.read_csv(obj['Body'], index_col=None)

    # Read feature_eng_data.csv
    obj2 = s3.get_object(Bucket='cloudengdfs', Key='feature_eng_data.csv')
    df1 = pd.read_csv(obj2['Body'], index_col=None)

    # Read stockx_data.csv
    obj3 = s3.get_object(Bucket='cloudengdfs', Key='stockx_data.csv')
    df3 = pd.read_csv(obj3['Body'], index_col=None)

    # Read cleaned_data.csv
    obj4 = s3.get_object(Bucket='cloudengdfs', Key='cleaned_data.csv')
    rec_data = pd.read_csv(obj4['Body'], index_col=None)


    return df, df1, df3, rec_data
df, df1, df3, rec_data = dataframes()


@st.cache_resource(ttl=1 * 24 * 60 * 60)
def load_models() -> Tuple:
    """
    Load the base models.

    Returns:
        tuple: A tuple of models.
    """
    #load base models
    s3 = boto3.client('s3')
    bucket_name = 'cloudengdfs'
    gb_object_name = 'model_objects/base_lin_model.pkl'
    gb_file_path = 'base_lin_model.pkl'
    s3.download_file(bucket_name, gb_object_name, gb_file_path)
    with open(gb_file_path, 'rb') as f:
        base_lin_model = pickle.load(f)

    bucket_name = 'cloudengdfs'
    gb_object_name = 'model_objects/base_gb_model.pkl'
    gb_file_path = 'base_gb_model.pkl'
    s3.download_file(bucket_name, gb_object_name, gb_file_path)
    with open(gb_file_path, 'rb') as f:
        base_gb_model = pickle.load(f)

    bucket_name = 'cloudengdfs'
    gb_object_name = 'model_objects/best_gb_model.pkl'
    gb_file_path = 'best_gb_model.pkl'
    s3.download_file(bucket_name, gb_object_name, gb_file_path)
    with open(gb_file_path, 'rb') as f:
        best_gb_model = pickle.load(f)

    return base_gb_model, base_lin_model, best_gb_model

base_gb_model, base_lin_model, best_gb_model = load_models()



data_project = st.sidebar.selectbox("Select part",
                                    ('Introduction', "App"))
title = '<p style="font-family:times-new-roman; color:White; font-size: 22px;">' \
            'Group: Yaasir Ahmed, Ishu Kalra, Siche Chen, Zijian Wang, Cameran Frank <br>' \
            'Project Title: Determining Fair Price For A Shoe For Consignment Stores<br> </p>'
if data_project == "Introduction":
    st.markdown(title,  unsafe_allow_html=True)

elif data_project == "App":
    models = df["Sneaker Name"].unique()
    st.markdown("Use the drop downs to find the latest price prediction for a sneaker model of your choice")
    brands = [" ", "Adidas", "Nike"]
    yeezy_shoes = df[df["Brand"] == "adidas"]["Sneaker Name"].unique()
    off_white_shoes = df[df["Brand"] == "nike"]["Sneaker Name"].unique()
    select_brand = st.selectbox("Select a Model",
                                brands)
    if select_brand == "Adidas":
        select_shoe = st.selectbox("Please select a type of shoe", yeezy_shoes)
        # select_date = st.selectbox("Select a timeframe", time_frame)
        st.write("Below is the most recent sales data for the selected shoe")
        filtered_data = df3[df3["Sneaker Name"] == select_shoe]
        st.dataframe(filtered_data, width=1000)

        X = rec_data.drop(['Last Sale Price', "Buyer Region"], axis=1)
        y = rec_data['Last Sale Price']

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        # gb_model = GradientBoostingRegressor().fit(X_train, y_train)
        y_pred = best_gb_model.predict(X)
        pred_df = pd.DataFrame(models)
        pred_df['Predicted price'] = y_pred
        pred_df = pred_df.rename(columns={0: 'Sneaker Name'})

        filtered_pred = pred_df[pred_df["Sneaker Name"] == select_shoe]
        st.write("Below is the predicted price for the shoe you have selected if you were to order it today")
        st.dataframe(filtered_pred, width=1000)


    elif select_brand == "Nike":
        select_shoe = st.selectbox("Please select a type of shoe", off_white_shoes)
        # select_date = st.selectbox("Select a timeframe", time_frame)
        filtered_data = df3[df3["Sneaker Name"] == select_shoe]
        st.dataframe(filtered_data)
        st.write("Below is the predicted price for the shoe you have selected if you were to order it today")
        X = rec_data.drop(['Last Sale Price', "Buyer Region"], axis=1)
        y = rec_data['Last Sale Price']

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = best_gb_model.predict(X)
        pred_df = pd.DataFrame(models)
        pred_df['Predicted price'] = y_pred
        pred_df = pred_df.rename(columns={0: 'Sneaker Name'})

        filtered_pred = pred_df[pred_df["Sneaker Name"] == select_shoe]
        st.write("Below is the predicted price for the shoe you have selected if you were to order it today")
        st.dataframe(filtered_pred, width=1000)

