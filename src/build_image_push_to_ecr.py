import base64
import docker
import boto3

# with many thanks to this guy:
# https://github.com/AlexIoannides/py-docker-aws-example-project/blob/master/deploy_to_aws.py

docker_client = docker.from_env()
ecr_client = boto3.client('ecr')
repository_name = 'future_generation'


ecr_credentials = ecr_client.get_authorization_token()['authorizationData'][0]
ecr_username = 'AWS'
ecr_password = (
    base64.b64decode(ecr_credentials['authorizationToken'])
    .replace(b'AWS:', b'')
    .decode('utf-8'))

ecr_url = ecr_credentials['proxyEndpoint']
# tag image for AWS ECR
ecr_repo_name = f"{ecr_url.replace('https://', '')}/{repository_name}"
assert ecr_repo_name == '714011423642.dkr.ecr.eu-west-1.amazonaws.com/future-generation'
print('ecr_repo_name', ecr_repo_name)

login_response = docker_client.login(username=ecr_username, password=ecr_password, registry=ecr_url)
print(login_response)

image, build_log = docker_client.images.build(path='..', rm=True)

# Tag your image with the Amazon ECR registry, repository, and optional image tag name combination to use.
# The registry format is aws_account_id.dkr.ecr.region.amazonaws.com.
# The repository name should match the repository that you created for your image.
# If you omit the image tag, we assume that the tag is latest.
image.tag(ecr_repo_name, tag='1')

# push image the ECR
push_log = docker_client.images.push(ecr_repo_name, tag='1')
print(push_log)

