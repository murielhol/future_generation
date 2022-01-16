# future_generation

# Docker
## build the container
`docker build -t future_generation .`

## to debug the container
`docker run -it future_generation /bin/bash`

To stop press ctrl-d


aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <container registry id>.dkr.ecr.us-east-1.amazonaws.com
docker tag docker_image_name:latest <container registry id>.dkr.ecr.us-east-1.amazonaws.com/docker_image_name:latest
docker push <container registry id>.dkr.ecr.us-east-1.amazonaws.com/docker_image_name:latest

# resources

[Stochastic wavenet (2018). Guokun Lai, Bohan Li, Guoqing Zheng, Yiming Yang](https://arxiv.org/abs/1806.06116)
[Alexioannides script to build image and push to ecr ](https://github.com/AlexIoannides/py-docker-aws-example-project/blob/master/deploy_to_aws.py)