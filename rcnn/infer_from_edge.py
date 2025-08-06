import boto3
import json
import numpy as np
from PIL import Image
import torchvision.transforms as T

# --- Configuration ---
endpoint_name = "your-endpoint-name"   # replace with your deployed endpoint name
region = "us-west-2"
image_path = "test_image.jpg"

# --- Load and preprocess image ---
transform = T.ToTensor()
img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)  # shape: (1, C, H, W)
payload = input_tensor.numpy().tolist()     # convert to list for JSON serialization

# --- Invoke SageMaker endpoint ---
sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

response = sm_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload)
)

# --- Parse prediction ---
result = json.loads(response["Body"].read().decode("utf-8"))
print("Inference result:", result)
