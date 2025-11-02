import os
import json
import base64
import io
import torch
import mlflow
import torchvision.transforms as transforms
from PIL import Image

# This function is called once when the service starts up.
def init():
    """Initializes the model and class names for the Condition model."""
    global model, class_names
    
    # Define the class names for the Condition model in the correct order (index 0, 1)
    class_names = ['good', 'worn']
    
    # AZUREML_MODEL_DIR is an environment variable pointing to the model folder.
    model_root_path = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_root_path, "model")
    
    # Load the MLflow-saved PyTorch model.
    model = mlflow.pytorch.load_model(model_path)
    model.eval()  # Set the model to evaluation mode

# This function is called for every invocation of the endpoint.
def run(raw_data):
    """Processes an image and returns a prediction for tyre condition."""
    try:
        # 1. Parse the incoming JSON request data
        data = json.loads(raw_data)
        image_base64 = data['image']
        
        # 2. Decode the base64 string to bytes and load into a PIL Image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # 3. Define and apply the image transformations for the Condition model
        # Based on your dataloaders.py ('tfm_eval' for 'condition').
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = transform(image).unsqueeze(0)

        # 4. Perform the prediction
        with torch.no_grad():
            output = model(input_tensor)
            
            # Apply Softmax to convert raw scores (logits) into probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get the top probability and its corresponding class index
            top_prob, top_idx = torch.max(probabilities, 1)
            
            predicted_class = class_names[top_idx.item()]
            confidence_score = top_prob.item()
            
            # Create a dictionary of all class probabilities
            all_probabilities = {class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}

        # 5. Return a detailed JSON response
        return {
            "prediction": predicted_class,
            "confidence": confidence_score,
            "probabilities": all_probabilities
        }
        
    except Exception as e:
        error = str(e)
        return {"error": error}