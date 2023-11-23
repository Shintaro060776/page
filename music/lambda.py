import boto3
import json
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    logger.info("Received event: " + json.dumps(event))
    try:
        endpoint_name = os.environ['SAGEMAKER_ENDPOINT_NAME']
    except KeyError:
        logger.error(
            "SAGEMAKER_ENDPOINT_NAME not set in environment variables")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": "SAGEMAKER_ENDPOINT_NAME not set in environment variables"})
        }

    runtime = boto3.client('sagemaker-runtime')

    try:
        if 'text' not in event:
            raise ValueError("No text key found in the body")

        text = event['text']
        data = json.dumps({"text": text}).encode('utf-8')
        logger.info(f"Sending data to SageMaker endpoint: {data}")

        response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                           ContentType='application/json',
                                           Body=data)
        result = json.loads(response['Body'].read().decode())
        logger.info(f"Received response from SageMaker endpoint: {result}")

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        logger.error(f"Error during SageMaker invocation: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                "error": str(e),
                "message": "An error occurred while processing the request."
            })
        }
