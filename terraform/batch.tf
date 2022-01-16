resource "aws_security_group" "default_security_group" {
  name        = var.name
  description = "Security group for ${var.name} acceptance tests."
  vpc_id      = data.aws_vpc.default.id
}


resource "aws_batch_compute_environment" "python_comp_env" {
  compute_environment_name = var.name

  compute_resources {
    instance_role = "arn_of_an_instance_role"
    allocation_strategy = "BEST_FIT"
    instance_type = ["optimal"]

    max_vcpus     = 10
    min_vcpus     = 0

    security_group_ids = [aws_security_group.default_security_group.id]
    subnets = [data.aws_vpc.default.id]
    type = "EC2"
  }

  service_role = "arn_of_aws_batch_service_role"
  type         = "MANAGED"
}

resource "aws_batch_job_queue" "python_scripts_queue" {
  name                 = "python_scripts_queue"
  state                = "ENABLED"
  priority             = 1
  compute_environments = [aws_batch_compute_environment.python_comp_env.arn]
}

resource "aws_batch_job_definition" "job-definition" {
  name                 = var.name
  type                 = "container"
//  container_properties = file("job_definition.json")
container_properties = <<CONTAINER_PROPERTIES
{
  "image": "${aws_ecr_repository.ecr_repo.arn}:latest",
  "jobRoleArn": "${aws_iam_role.job-role.arn}",
  "vcpus": 1,
  "memory": 1024,
  "command": ["run"]
}
CONTAINER_PROPERTIES
}


data "aws_iam_policy_document" "acceptance_test" {
  statement {
    actions = [
      "s3:GetObject*",
      "s3:PutObject*",
    ]

    resources = [
      aws_s3_bucket.output_bucker.arn
    ]
  }
}

# store results in s3
resource "aws_s3_bucket" "output_bucker" {
  bucket = "${var.name}-output"

  tags = {
    Component = var.name
    Environment = terraform.workspace
  }
}