import os
import json
import requests

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_ENDPOINT = "https://api.openai.com/v1/engines/davinci/completions"

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}


def alien_speak(text):
    translation_table = str.maketrans(
        "aeiou", "12345")
    return "👽: " + text.translate(translation_table)


def lambda_handler(event, context):
    try:
        prompt_text = event['body']

        data = {
            "prompt": prompt_text,
            "max_tokens": 150
        }

        response = requests.post(
            OPENAI_ENDPOINT, headers=headers, data=json.dumps(data))
        response_data = response.json()

        if response.status_code != 200:
            return {
                "statusCode": response.status_code,
                "body": json.dumps({"error": response_data.get("error", "Unknown error")})
            }

        alien_response = alien_speak(
            response_data["choices"][0]["text"].strip())
        return {
            "statusCode": 200,
            "body": json.dumps({"response": alien_response})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
