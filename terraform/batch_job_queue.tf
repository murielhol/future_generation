

resource "aws_batch_job_queue" "future_generation_training_queue" {
  name     = var.name
  state    = "ENABLED"
  priority = 1
  compute_environments = [
    aws_batch_compute_environment.future_generation_training.arn,
  ]
}