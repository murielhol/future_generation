# future_generation

# Docker
## build the container
`docker build -t future_generation .`

## to run the container on your local machine
`aws-vault exec private -- docker run -it -e AWS_REGION -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN -e AWS_SECURITY_TOKEN future_generation`

# terraform
I use aws-vault to handle credentials. 

Note that you need to setup your MFA device in order to be able to create the IAM roles for the batch job.


# resources
https://docs.aws.amazon.com/batch/latest/userguide/service_IAM_role.html

[Stochastic wavenet (2018). Guokun Lai, Bohan Li, Guoqing Zheng, Yiming Yang](https://arxiv.org/abs/1806.06116)
[Alexioannides script to build image and push to ecr ](https://github.com/AlexIoannides/py-docker-aws-example-project/blob/master/deploy_to_aws.py)