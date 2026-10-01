import os, boto3
from dotenv import load_dotenv
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
s3 = boto3.client("s3", region_name=AWS_REGION)

def upload_chunk_text(chunk_text: str, key: str) -> str:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not configured")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=chunk_text.encode("utf-8"))
    return f"s3://{S3_BUCKET}/{key}"

def download_chunk_text(key: str) -> str:
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return resp["Body"].read().decode("utf-8")
