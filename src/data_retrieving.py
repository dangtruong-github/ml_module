import logging
import os

from abc import ABC, abstractmethod
import pandas as pd

sql_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql_files")

class RetrieveData(ABC):
    """
    Abstract class for all data retrieving strategy: 
    - Retrieve 3 tables: movies, users, ratings
    """
    @abstractmethod
    def retrieve_data_sql(self, conn):
        """
        Args:
            conn: connection to postgresql database
        Returns:
            None
        """
        pass

class Baseline(RetrieveData):
    """
        Baseline strategy, or simplest strategy
        - Movie: get id + vote_average
        - User: get id of user who is not staff
        - Ratings: get everything
    """
    def retrieve_data_sql(self, conn):
        with open(os.path.join(sql_folder_path, "baseline", 'get_movies_id.sql'), 'r') as file:
            query_movie = file.read()
        df_movie = pd.read_sql_query(query_movie, conn)
        logging.info(f"Movie data: \n{df_movie}")

        with open(os.path.join(sql_folder_path, "baseline", 'get_users_id.sql'), 'r') as file:
            query_user = file.read()
        df_user = pd.read_sql_query(query_user, conn)
        logging.info(f"User data: \n{df_user}")

        with open(os.path.join(sql_folder_path, "baseline", 'get_ratings.sql'), 'r') as file:
            query_rating = file.read()
        df_rating = pd.read_sql_query(query_rating, conn)
        logging.info(f"Rating data: \n{df_rating}")

        conn.close()

        return df_movie, df_user, df_rating