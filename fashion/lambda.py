import boto3
import json
import logging
import base64

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    try:
        runtime = boto3.client('sagemaker-runtime')

        endpoint_name = os.environ['ENDPOINT_NAME']

        payload = json.dumps({'nz': 100})

        response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                           ContentType='application/json',
                                           Body=payload)

        result = json.loads(response['Body'].read().decode())

        encoded_image = base64.b64encode(result).decode('utf-8')

        return {
            'statusCode': 200,
            'body': json.dumps({'image': encoded_image})
        }

    except Exception as e:
        logger.error("エラーが発生しました", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps('エラーが発生しました')
        }
