resource "aws_api_gateway_rest_api" "blog_api" {
  name        = "BlogAPI"
  description = "API for a Blog App"
  body        = file("openapi_definition.json")
}

resource "aws_api_gateway_deployment" "blog_api_deployment" {
  rest_api_id = aws_api_gateway_rest_api.blog_api.id
  stage_name  = "prod"
  depends_on  = [aws_api_gateway_integration.blog_api_integration]
}

resource "aws_api_gateway_integration" "blog_api_integration" {
  rest_api_id = aws_api_gateway_rest_api.blog_api.id
  resource_id = aws_api_gateway_resource.blog_resource.id
  http_method = aws_api_gateway_method.blog_method.http_method

  type                    = "AWS_PROXY"
  integration_http_method = "POST"
  uri                     = aws_lambda_function.blog_lambda.invoke_arn
}

resource "aws_lambda_permission" "apigw_lambda_permission" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.blog_lambda.function_name
  principal     = "apigateway.amazonaws.com"
}