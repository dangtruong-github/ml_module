from pipelines.training_pipeline import pipeline
import logging

from zenml.client import Client

if __name__ == "__main__":
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    df = pipeline()
    logging.info(f"Predicted rating: \n{df}")
    #pass