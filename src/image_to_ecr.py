import base64
import docker
import boto3

# https://github.com/AlexIoannides/py-docker-aws-example-project/blob/master/deploy_to_aws.py


def build_image(tag, repository_name, docker_client):
    image, build_log = docker_client.images.build(path='.', tag={f"{repository_name}:{tag}"}, rm=True)
    return image


def push_image_to_ecr(image, tag, repository_name, docker_client, ecr_client):

    # Authenticate your Docker client to the Amazon ECR registry to which you intend to push your image.
    # Authentication tokens must be obtained for each registry used, and the tokens are valid for 12 hours.
    ecr_credentials = (ecr_client.get_authorization_token()['authorizationData'][0])
    ecr_username = 'AWS'

    ecr_password = (base64.b64decode(ecr_credentials['authorizationToken']).replace(b'AWS:', b'').decode('utf-8'))
    ecr_url = ecr_credentials['proxyEndpoint']

    # get Docker to login/authenticate with ECR
    docker_client.login(username=ecr_username, password=ecr_password, registry=ecr_url)

    # tag image for AWS ECR
    ecr_repo_name = '{}/{}'.format(ecr_url.replace('https://', ''), f"{repository_name}:{tag}")
    print('ecr_repo_name', ecr_repo_name)

    # Tag your image with the Amazon ECR registry, repository, and optional image tag name combination to use.
    # The registry format is aws_account_id.dkr.ecr.region.amazonaws.com.
    # The repository name should match the repository that you created for your image.
    # If you omit the image tag, we assume that the tag is latest.
    image.tag(ecr_repo_name, tag='latest')

    # push image the ECR
    push_log = docker_client.images.push(ecr_repo_name, tag='latest')


docker_client = docker.from_env()
ecr_client = boto3.client('ecr')

tag = 'latest'
repository_name = 'future_generation'


image = build_image(tag, repository_name, docker_client)
push_image_to_ecr(image, tag, repository_name, docker_client, ecr_client)