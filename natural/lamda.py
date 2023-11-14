import os
import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_runtime = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME')


def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(body)
        )

        result = json.loads(response['Body'].read().decode())

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        logger.error("Error during processing: %s", str(e), exc_info=True)

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
