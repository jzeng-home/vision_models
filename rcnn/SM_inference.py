# inference.py
import io, json, os, logging, traceback
import torch, torchvision
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = 2  # must match training

def get_model_instance_segmentation(num_classes):
    # IMPORTANT: never download weights in the endpoint container
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def model_fn(model_dir):
    try:
        logger.info("Torch versions: torch=%s, torchvision=%s", torch.__version__, torchvision.__version__)
        logger.info("model_dir contents: %s", os.listdir(model_dir))

        model = get_model_instance_segmentation(NUM_CLASSES)
        state_path = os.path.join(model_dir, "model.pth")

        # If you’re on PyTorch >=2.4 you *can* use weights_only=True safely for state_dicts:
        state = torch.load(state_path, map_location="cpu")  # or torch.load(state_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("load_state_dict: missing=%d, unexpected=%d", len(missing), len(unexpected))
        if missing: logger.info("Missing keys sample: %s", missing[:10])
        if unexpected: logger.info("Unexpected keys sample: %s", unexpected[:10])

        model.eval()
        return model
    except Exception as e:
        logger.error("model_fn crashed: %s", e)
        logger.error(traceback.format_exc())
        raise

def input_fn(request_body, content_type='application/x-image'):
    if content_type != 'application/x-image':
        raise ValueError(f"Unsupported content type: {content_type}")
    image = Image.open(io.BytesIO(request_body)).convert("RGB")
    x = F.to_tensor(image)  # [C,H,W] float32 in [0,1]
    if x.shape[0] > 3:
        x = x[:3, ...]
    return x

def predict_fn(input_data, model):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        x = input_data.to(device)
        with torch.no_grad():
            pred = model([x])[0]
        return {
            "boxes": pred["boxes"].detach().cpu().tolist(),
            "scores": pred["scores"].detach().cpu().tolist(),
            "labels": pred["labels"].detach().cpu().tolist(),
            # Add masks later once everything works
            # "masks": pred["masks"].detach().cpu().squeeze(1).tolist(),
        }
    except Exception as e:
        logger.error("predict_fn crashed: %s", e)
        logger.error(traceback.format_exc())
        # Surface error in logs; 500 to client is fine
        raise

def output_fn(prediction, accept='application/json'):
    if accept != 'application/json':
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction)