import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os

current_directory = os.path.dirname(os.path.abspath(__file__))

print(current_directory)

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for preprocess data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",          
            ], axis=1)

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])

            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(f"Error in divide data: {e}")
            raise e
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train, test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train, test
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data[["review_score"]]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            #y_train = pd.DataFrame(y_train, columns=["review_score"])
            #y_test = pd.DataFrame(y_test, columns=["review_score"])

            #logging.info(f"X_train: {type(X_train)}")
            #logging.info(f"X_test: {type(X_test)}")
            #logging.info(f"y_train: {type(y_train)}")
            #logging.info(f"y_test: {type(y_test)}")
            #logging.info(f"y_test: {y_test}")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e
        
class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
if __name__ == "__main__":
    parent_path = os.path.dirname(current_directory)
    data_path = os.path.join(parent_path, "data", "olist_customers_dataset.csv")
    
    data = pd.read_csv(data_path)
    data_cleaning = DataCleaning(data, DataPreprocessStrategy)
    data_cleaning.handle_data()
