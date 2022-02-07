
/* Compute environment, when using FARGATE you need the following:

1. A VPC with subnets. Note that the ECS need to be able to
   make API calls and so either your subnet needs to be public or
   it needs to have a NAT gateway

2. aws batch service role: the IAM role that allows AWS Batch
    to make calls to other AWS services on your behalve

*/


resource "aws_vpc" "batch_vpc" {
  cidr_block = "10.1.0.0/16"
}

resource "aws_subnet" "batch_subnet" {
  vpc_id     = aws_vpc.batch_vpc.id
  map_public_ip_on_launch = true
  cidr_block = "10.1.1.0/24"
}


resource "aws_security_group" "batch_security_group" {

  name = "aws_batch_compute_environment_security_group"

  vpc_id = aws_vpc.batch_vpc.id
  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
    description = "Allow egress."
  }
}


//data "aws_vpc" "default" {
//  default = true
//}
//
//data "aws_subnet_ids" "default" {
//  vpc_id = data.aws_vpc.default.id
//}
//
//data "aws_security_groups" "default" {
//  vpc_ids = [data.aws_vpc.default.id]
//}



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
    security_group_ids = [aws_security_group.batch_security_group.id]
    subnets = [aws_subnet.batch_subnet.id]
    type = "FARGATE_SPOT"
  }

  service_role = aws_iam_role.aws_batch_service_role.arn
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
  type         = "MANAGED"
  state = "ENABLED"

}