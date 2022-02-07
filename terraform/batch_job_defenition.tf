
/* Batch definition, when using FARGATE you need the following:

1. ecs task execution role: grants the Amazon ECS container and
   AWS Fargate agents permission to make AWS API calls on your behalf.
   Attach any additional policies if needed.


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

# attach another policy so that it can access S3
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
resource "aws_iam_policy" "s3_policy" {
  name = "batch_get_put_s3_role"
  policy = data.aws_iam_policy_document.s3_policy.json
}

resource "aws_iam_role_policy_attachment" "s3_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = aws_iam_policy.s3_policy.arn
}


# now describe the actual job
resource "aws_batch_job_definition" "future_generation_training_job" {
  name = var.name
  platform_capabilities = ["FARGATE"]
  type = "container"
  container_properties = <<CONTAINER_PROPERTIES
{
  "command": ["pipenv", "run", "python", "src/main.py"],
  "image": "${aws_ecr_repository.ecr_repo.repository_url}:latest",
  "fargatePlatformConfiguration": {
    "platformVersion": "LATEST"
  },
  "resourceRequirements": [
    {"type": "VCPU", "value": "0.5"},
    {"type": "MEMORY", "value": "2048"}
  ],
  "executionRoleArn": "${aws_iam_role.ecs_task_execution_role.arn}",
  "networkConfiguration": {
                    "assignPublicIp": "ENABLED"
                }

}
CONTAINER_PROPERTIES
}

