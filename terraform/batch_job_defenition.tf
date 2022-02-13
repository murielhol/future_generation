
/* Batch definition, when using FARGATE you need the following:

1. ecs task execution role: grants the Amazon ECS container and
   AWS Fargate agents permission to make AWS API calls on your behalf.

2. A role for your batch job itself, e.g. this job wants to read and
   write files from a specific S3 bucket

3. The aws batch job definition, you can if you want submit this with the
   AWS sdk if for example the image name changes, or the run command.
   But in this case I always use the same tag and command.

*/

# policy for FARGATE to execute ECS

resource "aws_iam_role" "ecs_task_execution_role" {
  name               = "tf_test_batch_exec_role"
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
}

data "aws_iam_policy_document" "assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# a policy so that you job can access the S3 bucket
data "aws_iam_policy_document" "assume_role_policy_s3" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "s3_policy" {
  statement {
    actions = [
          "s3:GetObject*",
          "s3:PutObject*",
          "s3:ListBucket*",
    ]

    resources = [
      aws_s3_bucket.batch_job_bucket.arn,
      "${aws_s3_bucket.batch_job_bucket.arn}/*"
    ]
  }
}

resource "aws_iam_role" "future_generation_s3" {
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy_s3.json
  description = "Role for the batch job to access S3."
  name = "${var.name}-job-role"
  tags = var.tags
  inline_policy {
    name = "AwsBatchS3Policy"
    policy = data.aws_iam_policy_document.s3_policy.json
  }
}


# now describe the actual job
resource "aws_batch_job_definition" "future_generation_training_job" {
  name = var.name
  platform_capabilities = ["FARGATE"]
  type = "container"
  container_properties = <<CONTAINER_PROPERTIES
{
  "command": ["pipenv", "run", "--", "python", "src/main.py", "--mode", "train"],
  "image": "${aws_ecr_repository.ecr_repo.repository_url}:1",
  "fargatePlatformConfiguration": {
    "platformVersion": "LATEST"
  },
  "resourceRequirements": [
    {"type": "VCPU", "value": "0.5"},
    {"type": "MEMORY", "value": "2048"}
  ],
  "executionRoleArn": "${aws_iam_role.ecs_task_execution_role.arn}",
  "jobRoleArn": "${aws_iam_role.future_generation_s3.arn}",
  "networkConfiguration": {
                    "assignPublicIp": "ENABLED"
                },
  "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "${var.name}",
                    "awslogs-region": "${var.region}",
                    "awslogs-stream-prefix": "${var.name}"
                }

            }

}
CONTAINER_PROPERTIES
}

resource "aws_cloudwatch_log_group" "future_generation" {
  name = var.name
  retention_in_days = 7
  tags = var.tags
}