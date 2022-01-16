terraform {
  required_version = "~> 0.14"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket = "future-generation-backend"
    key = "prod/terraform.tfstate"
    region = var.region
  }
}

provider "aws" {
  region = var.region
}