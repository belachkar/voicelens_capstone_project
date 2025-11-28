import os

import pandas as pd
from google.cloud import bigquery
from prefect import flow, task

# from sentiment_model import predict_sentiment
# from topic_model import predict_topic
# from ner_model import extract_entities

@task(name="Load Data")
def load_data_from_bq():
    """Reads the raw text from BigQuery."""
    query = "SELECT review_id, review_text FROM `your_project.your_dataset.standardized_data`"
    # Use pandas-gbq or simple bq client
    df = pd.read_gbq(query, project_id="your_project")
    return df

@task(name="Run Sentiment Model")
def run_sentiment_task(df):
    """Colleague 1's Work"""
    print("Running Sentiment Model...")
    # df['predicted_sentiment'] = df['review_text'].apply(your_sentiment_model)
    # DUMMY LOGIC
    df_out = df[['review_id']].copy()
    df_out['predicted_sentiment'] = "positive"
    df_out['sentiment_score'] = 0.95

    # Write to specific table
    df_out.to_gbq("your_dataset.preds_sentiment", project_id="your_project", if_exists="replace")
    print("Sentiment saved to BQ.")

@task(name="Run Topic Model")
def run_topic_task(df):
    """Colleague 2's Work"""
    print("Running Topic Model...")
    # DUMMY LOGIC
    df_out = df[['review_id']].copy()
    df_out['predicted_topic'] = "Delivery"

    # Write to specific table
    df_out.to_gbq("your_dataset.preds_topics", project_id="your_project", if_exists="replace")
    print("Topics saved to BQ.")

@task(name="Run NER Model")
def run_ner_task(df):
    """Colleague 3's Work"""
    print("Running NER Model...")
    # DUMMY LOGIC
    df_out = df[['review_id']].copy()
    df_out['extracted_entities'] = '[{"text": "battery", "label": "PRODUCT"}]'

    # Write to specific table
    df_out.to_gbq("your_dataset.preds_ner", project_id="your_project", if_exists="replace")
    print("NER saved to BQ.")

@flow(name="VoiceLens Training Pipeline")
def main_pipeline():
    # 1. Load Data
    data = load_data_from_bq()

    # 2. Run Models in Parallel (Prefect handles this)
    sent_future = run_sentiment_task.submit(data)
    topic_future = run_topic_task.submit(data)
    ner_future = run_ner_task.submit(data)

    # 3. Wait for completion
    sent_future.result()
    topic_future.result()
    ner_future.result()
    print("Pipeline Finished! Streamlit View is updated.")

if __name__ == "__main__":
    main_pipeline()
