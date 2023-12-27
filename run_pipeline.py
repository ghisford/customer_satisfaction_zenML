from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # Run the pipeline
    train_pipeline(data_path="/home/nzima/customer_satisfaction_zenML/data/olist_customer_dataset.csv")