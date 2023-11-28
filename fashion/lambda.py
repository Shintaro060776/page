import boto3
import json
import logging
import base64
import os
from PIL import Image
import numpy as np
import io

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')


def lambda_handler(event, context):
    try:
        runtime = boto3.client('sagemaker-runtime')
        endpoint_name = os.environ['ENDPOINT_NAME']

        payload = json.dumps({'nz': 100})

        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )

        result = json.loads(response['Body'].read().decode())
        image_array = np.array(result)

        logger.info(f"応答データの形状: {image_array.shape}")

        if image_array.ndim == 4 and image_array.shape[0] == 1:
            image_array = np.transpose(image_array[0], (1, 2, 0))

        if np.min(image_array) < 0:
            image_array = (image_array + 1) * 127.5
        elif np.max(image_array) <= 1.0:
            image_array = image_array * 255

        image = Image.fromarray(image_array.astype(np.uint8))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        s3_path = 'images/imagefromsagemaker/image.png'
        bucket_name = 'vhrthrtyergtcere'
        s3_client.put_object(Bucket=bucket_name, Key=s3_path,
                             Body=image_bytes, ContentType='image/png')

        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        return {
            'statusCode': 200,
            'body': json.dumps({'image': encoded_image})
        }

    except Exception as e:
        logger.error("エラーが発生しました", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps(f'エラーが発生しました: {e}')
        }
