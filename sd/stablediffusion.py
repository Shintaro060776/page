import base64
import os
import requests
from datetime import datetime

def generate_image_from_text(input_text):
    engine_id = "stable-diffusion-xl-1024-v1-0"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    api_key = os.getenv("STABILITY_API_KEY")

    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": input_text
                }
            ],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d_%H_%M')
    for i, image in enumerate(data["artifacts"]):
        with open(f"./out/{formatted_time}_{i}.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))
