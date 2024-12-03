import boto3
import pandas as pd
import io

# Initialize S3 client
s3 = boto3.client('s3')

# S3 bucket and data key
BUCKET_NAME = 'gp-ai'  # Replace with your bucket name
DATA_KEY = 'data/synthetic_log_dataset.csv'  # Replace with your data key

def read_csv_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body']
    df = pd.read_csv(io.BytesIO(content.read()))
    return df

if __name__ == "__main__":
    df = read_csv_from_s3(BUCKET_NAME, DATA_KEY)
    df.to_csv('synthetic_log_dataset.csv', index=False)
    print("Data downloaded successfully.")
