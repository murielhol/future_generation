import boto3

client = boto3.client('batch')

response = client.submit_job(
    jobDefinition='future-generation',
    jobName='test_job',
    jobQueue='future-generation',
    containerOverrides={
    },
)

print(response)