import json
import base64
import requests
import os
import boto3
import pyheif
import time
from datetime import datetime
from PIL import Image
from io import BytesIO


def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        base64_image = body['base64Image']
        decoded_image = base64.b64decode(base64_image)

        image_stream = BytesIO(decoded_image)
        image = Image.open(image_stream)

        content_type = 'image/png'

        if image.format == 'HEIC':
            heif_file = pyheif.read(image_stream)
            image = Image.frombytes(
                heif_file.mode,
                (heif_file.width, heif_file.height),
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            content_type = 'image/jpeg'
            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            decoded_image = image_stream.getvalue()
        elif image.format == 'JPEG':
            content_type = 'image/jpeg'
            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            decoded_image = image_stream.getvalue()

        api_key = os.environ['AILAB_API_KEY']

        response = requests.post(
            "https://www.ailabapi.com/api/image/effects/ai-anime-generator",
            files={
                'image': ('image.png', decoded_image, 'application/octet-stream')
            },
            data={
                'type': 'boy-21'
            },
            headers={
                'ailabapi-api-key': api_key
            },
            timeout=10
        )

        if response.status_code != 200:
            raise Exception(
                f"AILAB API returned {response.status_code}: {response.text}")

        task_id = response.json().get('task_id')
        if not task_id:
            raise Exception(
                "Invalid response from AILAB API: task_id missing.")

        for _ in range(20):
            task_response = requests.get(
                f"https://www.ailabapi.com/api/common/query-async-task-result?task_id={task_id}",
                headers={
                    'ailabapi-api-key': api_key
                },
                timeout=10
            )

            if task_response.status_code != 200:
                raise Exception(
                    f"AILAB Task API returned {task_response.status_code}: {task_response.text}")

            task_data = task_response.json()

            if task_data["task_status"] == 2:
                result_url = task_data.get("data", {}).get("result_url")
                if not result_url:
                    raise Exception(
                        "Invalid response from AILAB API: result_url missing.")
                converted_image = requests.get(result_url).content
                break
            elif task_data["task_status"] == 3:
                raise Exception("AILAB processing failed.")

            time.sleep(5)

        converted_base64_image = base64.b64encode(converted_image).decode()

        s3 = boto3.client('s3')
        file_name = 'animation_{}.png'.format(
            datetime.now().strftime('%Y%m%d%H%M'))
        bucket_name = 'vhrthrtyergtcere'

        s3.put_object(Bucket=bucket_name, Key=file_name,
                      Body=converted_image, ContentType=content_type)

        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"

        SLACK_TOKEN = 'xoxb-6079630516694-6146268918432-vA0c2u7Cyg5KI6HYxloOXlTP'
        SLACK_CHANNEL = 'C062WSVC36V'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {SLACK_TOKEN}'
        }

        payload = {
            'channel': SLACK_CHANNEL,
            'text': f'新しいアニメーション画像がアップロードされました: {s3_url}'
        }
        slack_response = requests.post(
            'https://slack.com/api/chat.postMessage', headers=headers, json=payload, timeout=10)
        slack_response.raise_for_status()

        return {
            'statusCode': 200,
            'body': json.dumps({'base64Image': converted_base64_image}),
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            }
        }

    except requests.Timeout:
        return {
            'statusCode': 408,
            'body': json.dumps({'error': 'External service timeout'}),
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            }
        }
