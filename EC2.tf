resource "aws_key_pair" "ec2_key" {
  key_name   = "new_ssh_key_name"
  public_key = file("~/.ssh/new_ssh_key.pub")
}

resource "aws_instance" "example" {
  ami           = "ami-044dbe71ee2d3c59e"
  instance_type = "t2.micro"
  key_name      = aws_key_pair.ec2_key.key_name
  subnet_id     = aws_subnet.public_1a.id

  vpc_security_group_ids = [aws_security_group.allow.id]

  tags = {
    Name = "EC2_Test11"
  }

  user_data = <<-EOF
    #!/bin/bash
    sudo yum update -y
    sudo amazon-linux-extras install nginx1
    sudo systemctl start nginx
    sudo systemctl enable nginx
    echo "Hello from nginx" > /usr/share/nginx/html/index.html
  EOF
}