variable "region" {
  description = "AWS region to create resources in"
  type  = string
  default = "eu-west-1"
}

variable "name" {
    description = "Name to use for resources"
    type = string
    default = "future-generation"
}

variable "tags" {
    description = "tags"
    type = map(string)

    default = {
        "managed_by" = "terraform"
    }
}
