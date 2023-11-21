import boto3
import json
import os


def lambda_handler(event, context):
    endpoint_name = os.environ('SAGEMAKER_ENDPOINT_NAME')

    runtime = boto3.client('sagemaker-runtime')

    try:
        data = json.dumps(event['data'])
        response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                           ContentType='application/json',
                                           Body=data)

        result = json.loads(response['Body'].read().decode())

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        print(e)
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
