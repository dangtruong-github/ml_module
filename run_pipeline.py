from pipelines.training_pipeline import train_pipeline

import os
from zenml.client import Client

current_directory = os.path.dirname(os.path.abspath(__file__))

print(current_directory)

data_path_cur = os.path.join(current_directory, "data", "olist_customers_dataset.csv")

if __name__ == "__main__":
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=os.path.join(current_directory, "data", "olist_customers_dataset.csv"))
    #pass