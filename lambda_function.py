import json
import os
import requests
import base64
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

def lambda_handler(event, context):
    try:
        # API GatewayからのJSON Bodyをパース
        body = json.loads(event['body'])
        input_text = body.get('input_text', '')

        # 画像生成
        generated_files = generate_image_from_text(input_text)
        
        # 正常に画像が生成された場合のレスポンス
        return {
            'statusCode': 200,
            'body': json.dumps({
                "message": "Image generated",
                "imagePaths": generated_files
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # CORS対応
            }
        }
    except Exception as e:
        # エラーが発生した場合のレスポンス
        return {
            'statusCode': 500,
            'body': json.dumps({
                "message": "An error occurred",
                "error": str(e)
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # CORS対応
            }
        }

def get_image_from_stability_ai(input_text):
    try:
        url = "https://api.stability.ai/v1/generation/{engine_id}/text-to-image"
        headers = {
            "Authorization": "Bearer YOUR_STABILITY_AI_API_KEY",
            "Content-Type": "application/json"
        }
        payload = {
            "input_text": input_text
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Ensure we received a 200 response
        response.raise_for_status()

        return response.content  # assuming API returns image data directly

    except requests.RequestException as e:
        raise Exception(f"Failed to fetch image from Stability AI: {str(e)}")