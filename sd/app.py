from flask import Flask, request, jsonify
import stablediffusion  

app = Flask(__name__)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    input_text = request.json.get('input_text', '')  
    stablediffusion.generate_image_from_text(input_text)
    return jsonify({"message": "Image generated"})

if __name__ == "__main__":
    app.run(debug=True)