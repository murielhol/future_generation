
resource "aws_s3_bucket" "batch_job_bucket" {
  bucket = "${var.name}-batch-job"
  tags = var.tags
}
