output "ecr_repository_name" {
  value = aws_ecr_repository.ecr_repo.name
}

output "repository_url" {
  value = aws_ecr_repository.ecr_repo.repository_url
}

output "vpc_id" {
  value = aws_vpc.batch_vpc.id
}

output "subnets" {
  value = aws_subnet.batch_subnet.id
}

output "security_group" {
  value = aws_security_group.batch_security_group.id
}

output "bucket" {
  value = aws_s3_bucket.batch_job_bucket.arn
}