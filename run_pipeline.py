from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/home/nzima/customer_satisfaction_zenML/data/olist_customer_dataset.csv")


# this line is for running the mlflow ui
# mlflow ui --backend-store-uri file:/home/nzima/.config/zenml/local_stores/7394d502-ccdd-4df5-8f60-376025e7a2f4/mlruns