terraform {
  required_version = ">= 0.12"

  backend "s3" {
    bucket = "vhrthrtyergtcere"
    key    = "terraform.tfstate"
    region = "ap-northeast-1"
  }
}

provider "aws" {
  region = "ap-northeast-1"
}

variable "DEPLOY_PUBLIC_KEY" {
  description = "The public key for deployment"
  type        = string
  default     = ""
}