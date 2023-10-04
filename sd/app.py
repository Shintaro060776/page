from flask import Flask, request, jsonify
import stablediffusion

app = Flask(__name__)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        input_text = request.json.get('input_text', '')
        generated_files = stablediffusion.generate_image_from_text(input_text)
        return jsonify({"message": "Image generated", "imagePaths": generated_files})
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)