
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import os


@dataclass
class S3Handler:
    boto3_session: Optional[boto3.Session] = field(init=False)
    s3_client: Optional[boto3.Session.client] = field(init=False)
    s3_bucket: Optional[str] = field(init=False)

    def __post_init__(self):
        self.boto3_session = boto3.Session()
        self.s3_client = self.boto3_session.client('s3')
        self.s3_bucket = f"future-generation-batch-job"

    def get_file_from_s3(self, file_name):
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=file_name)
        return response['Body']

    def download_file_from_s3(self, s3_file_name: str, local_file_name: str):
        try:
            self.s3_client.download_file(self.s3_bucket, s3_file_name, local_file_name)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

    def upload_file_to_s3(self, source_file_name: str, target_file_name: str):
        self.s3_client.upload_file(source_file_name, self.s3_bucket, target_file_name)

    def upload_outputs_to_s3(self, model_name):
        path = Path('outputs', model_name)
        for root, dirs, files in os.walk(path):
            for file in files:
                self.upload_file_to_s3(f"{root}/{file}", f"{root}/{file}")