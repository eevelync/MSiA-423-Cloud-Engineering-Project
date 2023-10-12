import os
import boto3
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
from PIL import Image
import logging
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data_project = st.sidebar.selectbox("Select part",
                                    ('Introduction', 'Data Engineering', 'EDA', 'ML Model Development',
                                     'Model Experimentation And Optimization','ML Pipeline and Automation',
                                     "AWS Usage and Cost", 'ML Model Deployment'))

new_title = '<p style="font-family:times-new-roman; color:White; font-size: 28px;">' \
            'Group: Yaasir Ahmed, Zijian Wang, Siche Chen, Ishu Kalra, Cameran Frank <br>' \
            'Project Title: Determining Fair Price For A Shoe For Consignment Stores<br> </p>'
s3 = boto3.client('s3')

logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


@st.cache_data(ttl=1 * 24 * 60 * 60)
def dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieves dataframes from S3 bucket and returns them.

    Returns:
        tuple: A tuple containing the following dataframes:
            - df: Original cleaned data
            - df1: Feature engineered original data
            - df3: Web Scraped StockX data
            - rec_data: Cleaned web scraped data
    """
    try:
        # Read original_cleaned_data.csv
        o1 = s3.get_object(Bucket="cloudengdfs", Key="StockX-Data-Contest-2019-3.csv")
        unc_df = pd.read_csv(o1["Body"], index_col=None)
        logging.info("Loaded 'original_cleaned_data.csv' successfully.")
    except Exception as e:
        logging.error("Error loading 'original_cleaned_data.csv': %s", str(e))
        raise

    try:
        # Read original_cleaned_data.csv
        obj = s3.get_object(Bucket='cloudengdfs', Key='original_cleaned_data.csv')
        df = pd.read_csv(obj['Body'], index_col=None)
        logging.info("Loaded 'original_cleaned_data.csv' successfully.")
    except Exception as e:
        logging.error("Error loading 'original_cleaned_data.csv': %s", str(e))
        raise

    try:
        # Read feature_eng_data.csv
        obj2 = s3.get_object(Bucket='cloudengdfs', Key='feature_eng_data.csv')
        df1 = pd.read_csv(obj2['Body'], index_col=None)
        logging.info("Loaded 'feature_eng_data.csv' successfully.")
    except Exception as e:
        logging.error("Error loading 'feature_eng_data.csv': %s", str(e))
        raise

    try:
        # Read stockx_data.csv
        obj3 = s3.get_object(Bucket='cloudengdfs', Key='stockx_data.csv')
        df3 = pd.read_csv(obj3['Body'], index_col=None)
        logging.info("Loaded 'stockx_data.csv' successfully.")
    except Exception as e:
        logging.error("Error loading 'stockx_data.csv': %s", str(e))
        raise

    try:
        # Read cleaned_data.csv
        obj4 = s3.get_object(Bucket='cloudengdfs', Key='cleaned_data.csv')
        rec_data = pd.read_csv(obj4['Body'], index_col=None)
        logging.info("Loaded 'cleaned_data.csv' successfully.")
    except Exception as e:
        logging.error("Error loading 'cleaned_data.csv': %s", str(e))
        raise

    return unc_df,df, df1, df3, rec_data



unc_df, df, df1, df3, rec_data = dataframes()
models = df["Sneaker Name"].unique()


@st.cache_data(ttl=1 * 24 * 60 * 60)
def eda_df() -> pd.DataFrame:
    """
    Performs exploratory data analysis on the dataframe.

    Returns:
        pd.DataFrame: A modified copy of the original dataframe with additional columns to perform EDA.
    """

    try:
        # Read original_cleaned_data.csv
        df4 = df.copy()
        df4['Order Date'] = pd.to_datetime(df4['Order Date'], format="%Y-%m-%d")
        df4['year'] = df4['Order Date'].dt.year
        logging.info("Performed exploratory data analysis successfully.")
    except Exception as e:
        logging.error("Error performing exploratory data analysis: %s", str(e))
        raise
    return df4


df4 = eda_df()

df4 = eda_df()

@st.cache_resource(ttl=1 * 24 * 60 * 60)
def load_models() -> Tuple[pickle.PickleBuffer, pickle.PickleBuffer, pickle.PickleBuffer]:
    """
    Loads pre-trained machine learning models from S3 bucket.

    Returns:
        tuple: A tuple containing the following models:
            - base_gb_model: Base gradient boosting model
            - base_lin_model: Base linear model
            - best_gb_model: Best gradient boosting model
    """
    try:
        # Load base models
        bucket_name = 'cloudengdfs'
        lin_object_name = 'model_objects/base_lin_model.pkl'
        lin_file_path = 'base_lin_model.pkl'
        s3.download_file(bucket_name, lin_object_name, lin_file_path)
        with open(lin_file_path, 'rb') as f:
            base_lin_model = pickle.load(f)
        logging.info("Loaded 'base_lin_model.pkl' successfully.")

        bucket_name = 'cloudengdfs'
        gb_object_name = 'model_objects/base_gb_model.pkl'
        gb_file_path = 'base_gb_model.pkl'
        s3.download_file(bucket_name, gb_object_name, gb_file_path)
        with open(gb_file_path, 'rb') as f:
            base_gb_model = pickle.load(f)
        logging.info("Loaded 'base_gb_model.pkl' successfully.")

        bucket_name = 'cloudengdfs'
        gb_object_name = 'model_objects/best_gb_model.pkl'
        gb_file_path = 'best_gb_model.pkl'
        s3.download_file(bucket_name, gb_object_name, gb_file_path)
        with open(gb_file_path, 'rb') as f:
            best_gb_model = pickle.load(f)
        logging.info("Loaded 'best_gb_model.pkl' successfully.")
    except Exception as e:
        logging.error("Error loading models: %s", str(e))
        raise

    return base_gb_model, base_lin_model, best_gb_model


base_gb_model, base_lin_model, best_gb_model = load_models()


if data_project == 'Introduction':
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(
        "For our project, we wanted to help consumers and consignmnet stores determine a fair sale or trade price for a shoe based on "
        "features such as latest sale date, size, etc... The Data we collected is from a Kaggle competition from 2017. The sneaker "
        "resell market is a seventy two billion dollar industry, with stockx having roughly twenty percent of that market share. "
        "In recent news, users have been moving away from buying directly from websites to buying in person from consignment stores, "
        "which allows them to negotiate a much fairer price compared to sites such as StockX, which usually adds on fees such as shipping, "
        "service fee, etc...")

elif data_project == "Data Engineering":
    st.subheader(" Data Ingestion")
    eng = ('''
import os
import boto3
s3 = boto3.resource(
    service_name = 's3'
    region_name = 'us-east-2')
obj = s3.Bucket("cloudengdfs").Object("StockX-Data-Contest-2019-3.csv").get()
df = pd.read_csv(obj['Body'], index_col = None)
    ''')
    st.write("- For data ingestion, the original data was read in from an s3 bucket into our python script."
             "Our columns include, the order date going from 2017 all the way to 2019, we have the brand, sneaker name,"
             "sale price of the shoe when bought, original retail price from the vendor, release date, shoe size, and "
             "the buyer region of the person who bought the shoe ")
    st.dataframe(unc_df.head())

    st.subheader("Pre-Processing")
    pre_proc = '''df['Sale Price'] = df['Sale Price'].str.replace('$', '').str.replace(',', '')
df['Retail Price'] = df['Retail Price'].str.replace('$', '').str.replace(',', '')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Release Date'] = pd.to_datetime(df['Release Date'])
df[['Sale Price', 'Retail Price']] = df[['Sale Price', 'Retail Price']].astype(int)
    '''
    st.write("- For pre-processing, first we wanted to get the dates in proper date time format, and in order"
             "to match the output for webscraping later on we changed the unique brands to either ('adidas' or 'nike') "
             "Following that, we converted the sale and retail price to integer formats in order to calculate EDA "
             "metrics")
    st.dataframe(df.head())

    de_proc = ["", "Data Ingestion", "Pre-Processing"]
    select_sec = st.selectbox("Select a code section to explore", de_proc)
    if select_sec == "Data Ingestion":
        st.code(eng, language = 'python')
    elif select_sec == "Pre-Processing":
        st.code(pre_proc, language='python')


elif data_project == "EDA":
    st.title("Exploratory Data Analysis")


    def graph1():
        vals = df.groupby("Brand").agg({"Brand": "count"})
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(vals.index, vals["Brand"])
        ax.set_ylabel("# of releases")
        sns.despine(ax=ax)
        st.pyplot(fig)
    def graph2():
        avg_rp = df.groupby("Brand").agg({"Retail Price": "mean"})
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(avg_rp.index, avg_rp["Retail Price"])
        sns.despine(ax=ax)
        ax.set_ylabel("Average retail price")
        st.pyplot(fig)

    def graph3():
        avg_resp = df.groupby("Brand").agg({"Sale Price": "mean"})
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(avg_resp.index, avg_resp["Sale Price"])
        sns.despine(ax=ax)
        ax.set_ylabel("Average Resell Price")
        st.pyplot(fig)

    def graph4():
        yeezy = df[df["Brand"] == 'adidas']
        offwhite = df[df["Brand"] == "nike"]
        yavg_sizerp = yeezy.groupby("Shoe Size").agg({"Sale Price": "mean"})
        oavg_sizerp = offwhite.groupby("Shoe Size").agg({"Sale Price": "mean"})

        fig, ax = plt.subplots(1, 2, figsize=(12, 7))
        ax[0].bar(yavg_sizerp.index, yavg_sizerp["Sale Price"])
        ax[0].set_ylabel("Average Resell Price (adidas)", fontsize=12)
        ax[0].set_xlabel("Size")
        ax[1].bar(oavg_sizerp.index, oavg_sizerp["Sale Price"])
        ax[1].set_ylabel("Average Resell Price (nike)", fontsize=12)
        ax[1].set_xlabel("Size")
        for a in ax:
            sns.despine(ax=a)
            a.set_title("Average Resell Price Of Brands 2017-2019")
        st.pyplot(fig)

    def graph5():
        states_spend = df.groupby("Buyer Region").agg({"Sale Price": "sum"})
        states_dict = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
                       'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
                       'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
                       'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                       'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
        states_spend = df.groupby("Buyer Region").agg({"Sale Price": "sum"})
        color_scale = px.colors.sequential.Plasma
        fig = px.choropleth(locations=states_dict, locationmode="USA-states",
                            color=states_spend["Sale Price"], scope="usa", color_continuous_scale=color_scale)
        st.write("The state with the most spending on resell shoes is California, followed by New York, Oregon, "
                 "Texas, and Florida")

        st.plotly_chart(fig)
    def graph6():
        select_models = st.selectbox("Select a model to explore stability and average prices",
                                     models)
        # Filter the DataFrame based on a specific year
        desired_year = df4['year'].unique()
        select_year = st.selectbox("Select a year to explore:", desired_year)

        g6_df = df4[(df4['year'] == (select_year)) & (df4["Sneaker Name"] == select_models)]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.despine(ax=ax)
        ax.plot(g6_df["Order Date"], g6_df["Sale Price"])
        ax.axhline(g6_df["Sale Price"].mean(), linestyle="--", color='black', linewidth=4)
        ax.set_ylabel("Price", fontsize=15)
        ax.set_xlabel("Date", fontsize=15)
        st.pyplot(fig)


    graph_options = ["Which brand released more shoes in the years 2017-2019?",
                     "Which brand has the higher average retail price?",
                     "Which brand has the higher average resell price?",
                     "Does size have an effect on resell price?",
                     "Which region (State) spends the most on non-retail shoes?",
                     "How stable are resale prices?"]
    select_graph = st.selectbox("Select a question to answer: ", graph_options)
    if select_graph == "Which brand released more shoes in the years 2017-2019?":
        graph1()
        st.write("- Here we can see that adidas released over double the shoes that nike did in that time-frame")
    elif select_graph == "Which brand has the higher average retail price?":
        graph2()
        st.write("- On average adidas has higher retail prices compared to nike")
    elif select_graph == "Which brand has the higher average resell price?":
        graph3()
        st.write("- However when it comes to average resell prices, nike resells for almost double the adidas shoes do")
    elif select_graph == "Does size have an effect on resell price?":
        graph4()
        st.write("- Here we can see that shoe sizes don't really have an effect on resell prices, as they "
                 "are stable through out the low sizes up to around the more rare and harder to come by sizes "
                 "such as sizes 16 through 18")
    elif select_graph == "Which region (State) spends the most on non-retail shoes?":
        graph5()
    elif select_graph == "How stable are resale prices?":
        graph6()
        st.write("- In order to explore how stable prices, we added drop downs for the years the shoes were "
                 "sold, and the the specific models. Based on the graphs, we do see a lot of spikes indicating"
                 "different highs and lows, but the prices generally settle around the mean")

elif data_project == "ML Model Development":

    st.subheader("Feature Engineering")
    st.write("- For feature engineering, we first wanted to create dummies for both the brands, "
             "and all of the shoe models, then we calculated the markup by subtracting the resell "
             "price from the retail price and dividing by the retail price. Following that we "
             "calculated the age in days of the shoe by subtracting the order date from the release date,"
             "the mean resale price of the model. Lastly the mean resale price by model is simply just the average "
             "sale price for each model across time. ")
    st.dataframe(df1.head(), width = 1000)

    pr_models = ['Base Linear', "Base Xgboost"]
    select_models = st.selectbox("Select a model to explore", pr_models)
    X = df1.drop(['Sale Price', "Buyer Region"], axis=1)
    y = df1['Sale Price']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if select_models == "Base Linear":
     # Evaluate the model on the validation set
        y_pred = base_lin_model.predict(X_val)
        mse = round(mean_squared_error(y_val, y_pred), 2)
        accuracy = base_lin_model.score(X_train, y_train)
        residuals = y_pred - y_val
        signs = round(np.sign(residuals).mean()*100, 2)
        st.write("- Mean Squared Error", mse)
        st.write("- Model Accuracy On Test Data", accuracy)
        st.write("- On average is the model overestimating/underestimating and to what extent? ", signs)
        fig, ax = plt.subplots(figsize = (12, 7))

    elif select_models == "Base Xgboost":
        y_pred = base_gb_model.predict(X_val)
        accuracy = base_gb_model.score(X_train, y_train)
        mse1 = mean_squared_error(y_val, y_pred)

        residuals = y_pred - y_val
        signs = round(np.sign(residuals).mean()*100, 2)

        st.write("Mean Squared Error: ", mse1)
        st.write("Model Accuracy on Test Data: ", accuracy)

        st.write("On average is the model overestimating/underestimating and to what extent? ", signs)

        fig, ax = plt.subplots(figsize=(12, 7))
        df6 = pd.DataFrame([X.columns, base_gb_model.feature_importances_])
        df6 = df6.transpose()
        df6 = pd.concat([df6.head(4), df6.tail(3)], axis=0)
        ax.bar(df6[0], df6[1])
        plt.xticks(rotation=90)
        ax.set_ylabel("Importance", fontsize=12)
        ax.set_title("Feature Importance")
        sns.despine(ax=ax)
        st.pyplot(fig)


elif data_project == "Model Experimentation And Optimization":
    st.subheader("Model Experimentation and optimization")
    st.write("- Going back to the previous model results we can see that while they may perform "
             "similar in terms of accuracy gradient boosting is clearly the better option, with a lower MSE, and ,"
             "better residuals indicating that it isn't overestimating as much compared to "
             "basic linear regression")
    st.write("- Furthermore, since we implemented gradient boosting using default parameters, we wanted to "
             "perform GridSearchCV in order to find the optimal parameters to maximize the models performance. "
             "Following that below are the results of the parameter tuning")
    X = df1.drop(['Sale Price', "Buyer Region"], axis=1)
    y = df1['Sale Price']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = best_gb_model.predict(X_val)
    accuracy = best_gb_model.score(X_train, y_train)
    mse_bg = mean_squared_error(y_val, y_pred)

    residuals = y_pred - y_val
    signs = np.sign(residuals).mean() * 100

    st.write("Optimized Xgboost Model")
    st.write("Mean Squared Error: ", mse_bg)
    st.write("Model Accuracy on Test Data: ", accuracy)

    st.write("On average is the model overestimating/underestimating and to what extent? ", signs)
    st.write("- Strangely enough, even though the MSE for the optimized model is significantly lower compared"
             "to the base xgboost model, it is performing worse when it comes to over estimating the "
             "validated sale price")

elif data_project == "ML Pipeline and Automation":
    models = df['Sneaker Name'].unique()
    st.subheader("Webscraping")
    st.write("- Since sales data is always updating, we decided to automate the webscraping functions using a lambda function, and an "
             "event bridge trigger that runs every five days. Once the csv file is updated to an s3 bucket that also sets off another event bridge "
             "trigger that runs the data cleaning lambda function, which prepares the new scraped data for new weekly predictions. "
             "In order to apply the machine learning techniques brought up previously, feature engineering was also applied to get "
             "the data in the same format.")
    def stockx_connection():
        web_code = '''def search(query, item):
        url = f'https://stockx.com/api/browse?_search={query}'

        ###bot protection
        headers = {
        'accept': 'application/json',
        'accept-encoding': 'utf-8',
        'accept-language': 'en-GB,en;q=0.9',
        'app-platform': 'Iron',
        'referer': 'https://stockx.com/en-gb',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.62 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
                    }

        html = requests.get(url=url, headers=headers)
        output = json.loads(html.text)
        return output['Products'][1]['market'][item]'''

        st.code(web_code, language='python')
        st.write("Using the connection we want to retrieve the latest ask, last sale, size, and model, and build the webscraped data frame"
                 "into the same format as the original data frame")

    def variables():
        df_code = '''def dataframe(model):
    em_list = []
    if i == model:
    em_list.append(float(search(model, "lastSale")
    em_list.append(float(search(model, "lastSaleSize")
    return em_list
        
def f_df(model)
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
    return p_df'''
        st.code(df_code, language='python')

    select_code = ["", "Webscraping"]
    web_drop = st.selectbox("Click on dropdown to view web scraping code", select_code)
    if web_drop == "Webscraping":
        stockx_connection()
        variables()
    st.dataframe(df3.head(), width = 1000)

    st.write("After finalizing the machine learning, webscraping we moved on to our final piece which is what the end "
             "user sees and interacts with")

elif data_project == "AWS Usage and Cost":
    st.subheader("Below is an image of the aws tools we used and how we configured them, with the cost associated with them")
    drop_downs = ["", "AWS Usage", "AWS Cost"]
    select = st.selectbox("Select a section to explore", drop_downs)

    if select == "AWS Usage":
        image = Image.open("aws_usage.jpg")
        st.image(image, use_column_width=True)
    elif select == "AWS Cost":
        image = Image.open("aws_cost.jpg")
        st.image(image)

elif data_project == "ML Model Deployment":
    st.subheader("Model Deployment")
    st.write("- Since the end user wouldn't care about the development and all the background stuff, "
             "we deployed the resulting weekly predictions into its own separate link to allow "
             "for easy interaction.")
    st.write("- In order to make predictions for the user is they were to order the shoe today, we changed the order "
             "date column to dt.today() to show predictions for todays prices")
    brands = [" ", "Adidas", "Nike"]
    yeezy_shoes = df[df["Brand"] == "adidas"]["Sneaker Name"].unique()
    off_white_shoes = df[df["Brand"] == "nike"]["Sneaker Name"].unique()
    st.subheader("Demo")
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
        y_pred = best_gb_model.predict(X)
        pred_df = pd.DataFrame(models)
        pred_df['Predicted price'] = y_pred
        pred_df = pred_df.rename(columns={0: 'Sneaker Name'})
        filtered_pred = pred_df[pred_df["Sneaker Name"] == select_shoe]
        st.write("Below is the predicted price for the shoe you have selected if you were to order it today")
        st.dataframe(filtered_pred, width=1000)

    elif select_brand == "Nike":
        select_shoe = st.selectbox("Please select a type of shoe", off_white_shoes)
        filtered_data = df3[df3["Sneaker Name"] == select_shoe]
        st.dataframe(filtered_data)
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

    st.write("Deployed App: [link](http://demo-app-1129752538.us-east-2.elb.amazonaws.com/)")

    st.write("Web Scraping Source: [link](https://github.com/yasserqureshi1/stockx-discord-bot/blob/master/src/stockx.py)")
    st.write("Kaggle Data Source:[link](https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest?resource=download)")
    st.subheader("Bugs/Improvements")
    st.write("Making too many requests can get your ip address blocked for about 24-48 hours,"
             "In order to get past this issue, we used a for loop with a 5 second time delay to mimic human interaction, "
             "and then set the automation to execute the loop once every five days as to avoid getting flagged. "
             "This automation process the uploads the data to the s3 and other lambda functions take over to do "
             "data cleaning and pre-processing. Ideally we would like to have a larger database of different shoes, brands, etc..."
             "Lastly, instead of drop downs, we thought of adding a search feature, however that isn't really viable unless "
             "we have a much larger database of shoes.")

    st.subheader("The End")






