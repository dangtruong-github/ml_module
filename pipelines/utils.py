import logging

import pandas as pd

from src.data_cleaning import DataCleaning, DataPreprocessStrategy

import os

current_directory = os.path.dirname(os.path.abspath(__file__))

parent_path = os.path.dirname(current_directory)
data_path_cur = os.path.join(parent_path, "data", "olist_customers_dataset.csv")

def get_data_for_test():
    try:
        df = pd.read_csv(data_path_cur)
        df = df.sample(n=100)
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e