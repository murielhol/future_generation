
/* Compute environment, when using FARGATE you need the following:

1. A VPC with subnets. Note that the ECS need to be able to
   make API calls and so either you need a NAT gateway
   or your subnets need to be public and your vpc
   needs an internet gateway

2. aws batch service role: the IAM role that allows AWS Batch
    to make calls to other AWS services on your behalve

*/

data "aws_vpc" "default" {
  default = true
}

data "aws_subnet_ids" default {
    vpc_id = data.aws_vpc.default.id
}

data "aws_security_group" default {
  vpc_id = data.aws_vpc.default.id
  name = "default"
}


resource "aws_iam_role" "aws_batch_service_role" {
  name = "aws_batch_service_role"

  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
    {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
        "Service": "batch.amazonaws.com"
        }
    }
    ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "aws_batch_service_role" {
  role       = aws_iam_role.aws_batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}


resource "aws_batch_compute_environment" "future_generation_training" {
  compute_environment_name = var.name

  compute_resources {
    max_vcpus     = 2
    security_group_ids = [data.aws_security_group.default.id]
    subnets = data.aws_subnet_ids.default.ids
    type = "FARGATE_SPOT"
  }

  service_role = aws_iam_role.aws_batch_service_role.arn
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
  type         = "MANAGED"
  state = "ENABLED"
}