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