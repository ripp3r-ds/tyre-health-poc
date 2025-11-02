import os
import requests
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="Tyre Health Check API")


origins = [
    "http://127.0.0.1:5500", 
    "http://localhost:5500",

]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


AML_ENDPOINT = os.getenv("AML_ENDPOINT")
AML_KEY = os.getenv("AML_KEY")



def call_azure_ml_endpoint(endpoint_url: str, api_key: str, image_bytes: bytes, deployment_type: str = None):
    """Helper function to call a deployed Azure ML model."""
    if not endpoint_url or not api_key:
        raise HTTPException(status_code=500, detail="Azure ML endpoint is not configured in the environment.")

    encoded_image = base64.b64encode(image_bytes).decode("ascii")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    if deployment_type:
        headers["azureml-model-deployment"] = deployment_type

    data = {
        "image": encoded_image
    }

    try:
        response = requests.post(endpoint_url, headers=headers, json=data)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error calling Azure ML service: {e}")

@app.post("/predict/condition")
async def predict_condition(file: UploadFile = File(...)):
    """Receives an image, sends it to the condition model, and returns the result."""
    image_bytes = await file.read()
    result = call_azure_ml_endpoint(AML_ENDPOINT, AML_KEY, image_bytes)
    return result

@app.post("/predict/pressure")
async def predict_pressure(file: UploadFile = File(...)):
    """Receives an image, sends it to the pressure model, and returns the result."""
    image_bytes = await file.read()
    result = call_azure_ml_endpoint(AML_ENDPOINT, AML_KEY, image_bytes, deployment_type="pressure-v1")
    return result

@app.get("/")
def read_root():
    """A simple root endpoint to check if the API is running."""
    return {"message": "Welcome to the Tyre Health Check API!"}