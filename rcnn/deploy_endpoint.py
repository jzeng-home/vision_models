import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
import io
import json

def model_fn(model_dir):
    # Load model architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        tensor = torch.tensor(data)
        return tensor
    elif request_content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        transform = T.ToTensor()
        return transform(image)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        prediction = model([input_data])[0]
    return prediction

def output_fn(prediction, response_content_type):
    return json.dumps({
        "boxes": prediction["boxes"].tolist(),
        "labels": prediction["labels"].tolist(),
        "scores": prediction["scores"].tolist()
    })


## Create a Model from the Estimator

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",  # use ml.g4dn.xlarge for GPU
    entry_point="inference.py",
    source_dir=".",  # directory with inference.py
    framework_version="1.13",
    py_version="py38"
)

##Invoke the Endpoint

## test deployment
from PIL import Image
import torchvision.transforms as T

transform = T.ToTensor()
img = Image.open("sample.png").convert("RGB")
tensor = transform(img)

result = predictor.predict(tensor.numpy())
print(result)

## Clean Up (Stop Billing)
predictor.delete_endpoint()