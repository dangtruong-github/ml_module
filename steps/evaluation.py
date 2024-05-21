import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import step
from zenml.client import Client
import mlflow
from sklearn.base import RegressorMixin

from src.evaluation import MSE, R2, RMSE

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse_score"]
]:
    """
    Trains the model on ingested data

    Args:
        df: the ingested data
        
    """
    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse_score = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse_score)
        
        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2_score)
        
        rmse_class = RMSE()
        rmse_score = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse_score)

        return r2_score, rmse_score
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e