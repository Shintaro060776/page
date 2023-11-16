import os
import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_runtime = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME')


def lambda_handler(event, context):
    body = event.get('body')

    if not body:
        logger.error("Empty request body")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Empty request body'})
        }

    try:
        logger.info(f"Received request body: {body}")

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(body)
        )

        result = json.loads(response['Body'].read().decode())

        if 'joke' not in result:
            raise ValueError(
                "Expected field 'joke' is missing in the response")

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        status_code = 500
        if isinstance(e, ValueError):
            status_code = 400
        return {
            'statusCode': status_code,
            'body': json.dumps({'error': str(e)})
        }
