import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from 'data_path'
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from 'data_path'
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from 'data_path'

    Args:
        data_path: path to the data
    Return:
        pd.DataFrame: ingested data
    """
    try:
        ingest_data_class = IngestData(data_path)
        df = ingest_data_class.get_data()

        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e