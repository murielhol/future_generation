output "ecr_repository_name" {
  value = aws_ecr_repository.ecr_repo.name
}

output "repository_url" {
  value = aws_ecr_repository.ecr_repo.repository_url
}

output "bucket" {
  value = aws_s3_bucket.batch_job_bucket.arn
}