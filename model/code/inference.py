import torch
import torch.nn as nn
import json
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PathMNISTCNN(nn.Module):
    def __init__(self):
        super(PathMNISTCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 9)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

def model_fn(model_dir):
    logger.info("Starting model loading from %s", model_dir)
    try:
        model = PathMNISTCNN()
        
        state_dict = torch.load(f"{model_dir}/best_model.pth", map_location="cpu")
        
        # Load the state dict into the model
        model.load_state_dict(state_dict)
        
        model.eval()
        
        logger.info("Model loaded successfully!!")
        return model
        
    except Exception as e:
        logger.error(f"Error in model_fn: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    try:
        logger.info(f"Got request with content type: {request_content_type}")
        input_data = json.loads(request_body)
        input_tensor = torch.FloatTensor(input_data)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        logger.info(f"Done processed input shape: {input_tensor.shape}")
        return input_tensor
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise

def predict_fn(input_tensor, model):
    try:
        logger.info("Starting prediction")
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = output.max(1)
        result = predicted.cpu().numpy().tolist()
        logger.info(f"Prediction res: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise

def output_fn(prediction, response_content_type):
    try:
        return json.dumps(prediction)
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise