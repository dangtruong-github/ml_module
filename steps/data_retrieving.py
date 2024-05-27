from zenml import step, pipeline
import psycopg2
from typing import Tuple
from typing_extensions import Annotated

import logging
import os
from dotenv import load_dotenv

import pandas as pd

from src.data_retrieving import Baseline

load_dotenv()

sql_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql_files")

@step(enable_cache=False)
def retrieve_data() -> Tuple[
    Annotated[pd.DataFrame, "df_movie"],
    Annotated[pd.DataFrame, "df_user"],
    Annotated[pd.DataFrame, "df_rating"],
]:
    conn = psycopg2.connect(
        dbname=os.environ.get('DB_NAME'),
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASSWORD'),
        host=os.environ.get('DB_HOST'),
        port=os.environ.get("DB_PORT")
    )

    data_retrieve_class = Baseline()
    df_movie, df_user, df_rating = data_retrieve_class.retrieve_data_sql(conn)

    conn.close()

    return df_movie, df_user, df_rating

@pipeline
def postgres_pipeline():
    connect_to_postgres()

if __name__ == "__main__":
    postgres_pipeline()