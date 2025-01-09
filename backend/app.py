from flask import Flask, request, jsonify
import boto3
import json
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure AWS using env vars
runtime = boto3.client('sagemaker-runtime',
    region_name=os.getenv('AWS_REGION')
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = Image.open(file).convert('RGB')
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image)
        image_array = image_tensor.numpy()
        
        response = runtime.invoke_endpoint(
            EndpointName=os.getenv('ENDPOINT_NAME'),
            ContentType='application/json',
            Body=json.dumps(image_array.tolist())
        )
        
        prediction = json.loads(response['Body'].read().decode())
        return jsonify({'prediction': prediction[0]})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)