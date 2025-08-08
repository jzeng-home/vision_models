# client_infer_with_masks.py
import io
import json
import base64
import boto3
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# ---------- CONFIG ----------
REGION = "us-west-2"
ENDPOINT_NAME = "pytorch-inference-2025-08-08-03-11-48-749"  # change if you re-deploy

IMAGE_PATH = "data/PennFudanPed/PNGImages/FudanPed00046.png"
SCORE_THRESHOLD = 0
MASK_THRESHOLD = 0.7
OUT_IMAGE = "prediction_vis.png"
# ----------------------------


def load_image_bytes(path):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def invoke_endpoint(image_bytes):
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/x-image",
        Body=image_bytes,
        Accept="application/json",
    )
    body = resp["Body"].read().decode("utf-8")
    return json.loads(body)

def _to_uint8_rgb_tensor(img_tensor):
    """
    Make image uint8 0..255 and force 3 channels (drop alpha).
    """
    # Normalize to 0..255 then cast to uint8 (matches your code path)
    img = (255.0 * (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())).to(torch.uint8)
    if img.size(0) > 3:
        img = img[:3, ...]
    return img

def _prepare_masks(prediction, H, W, mask_threshold=0.7):
    """
    Accept masks in one of these endpoint formats and return a torch.bool tensor [N,H,W]:
      1) prediction["masks"] as nested lists -> shape [N,H,W] floats 0..1
      2) prediction["masks"] as nested lists -> shape [N,H,W] ints {0,1}
      3) prediction["masks_b64"] as list of per-mask PNGs (grayscale) base64-encoded
    """
    masks = None

    if "masks" in prediction:
        m = torch.tensor(prediction["masks"])
        # m shape could be [N,H,W] floats or ints
        if m.dtype != torch.bool:
            # If float probs, threshold; if ints 0/1, this still works
            m = (m >= mask_threshold)
        masks = m.bool()

    elif "masks_b64" in prediction:
        bin_masks = []
        for b64png in prediction["masks_b64"]:
            png_bytes = base64.b64decode(b64png)
            pil = Image.open(io.BytesIO(png_bytes)).convert("L")  # grayscale
            pil = pil.resize((W, H))  # ensure same size as image if needed
            t = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(pil.tobytes()))
                 .numpy().reshape(H, W))
            )  # uint8 [H,W]
            bin_masks.append(t >= int(mask_threshold * 255))
        if bin_masks:
            masks = torch.stack(bin_masks, dim=0).bool()

    return masks  # torch.bool [N,H,W] or None

def visualize(image_path, prediction, score_threshold=0.5, mask_threshold=0.7, out_path="prediction_vis.png"):
    # Load original as tensor [C,H,W], dtype=uint8 0..255 (we'll normalize like your code)
    raw = read_image(image_path)  # this returns uint8 already, but we'll follow your pipeline
    H, W = raw.shape[-2], raw.shape[-1]

    # Parse predictions
    boxes = torch.tensor(prediction.get("boxes", []), dtype=torch.float32)
    scores = torch.tensor(prediction.get("scores", []), dtype=torch.float32)
    labels = prediction.get("labels", [])

    # Filter by score
    if boxes.numel() == 0:
        print("No detections returned.")
        img_for_draw = _to_uint8_rgb_tensor(raw)
        # show and save
        plt.figure(figsize=(12, 12))
        plt.imshow(img_for_draw.permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved visualization -> {out_path}")
        return

    keep = scores >= score_threshold
    if keep.any():
        boxes = boxes[keep]
        scores = scores[keep]
        labels = [l for k, l in zip(keep.tolist(), labels) if k]
    else:
        print("No detections above threshold.")
        img_for_draw = _to_uint8_rgb_tensor(raw)
        plt.figure(figsize=(12, 12))
        plt.imshow(img_for_draw.permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved visualization -> {out_path}")
        return

    # Your label format
    pred_labels = [f"pedestrian: {s:.3f}" for s in scores.tolist()]
    pred_boxes = boxes.long()

    # Prepare image like your code (normalize 0..255 and drop alpha)
    image = _to_uint8_rgb_tensor(raw)

    # Draw boxes
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    # Handle masks if present
    masks = _prepare_masks(prediction, H, W, mask_threshold)
    if masks is not None:
        # If masks were filtered by score, keep the same subset
        if masks.size(0) != pred_boxes.size(0):
            # Try to subset if we can infer original keep indices
            # Otherwise assume masks align with filtered boxes:
            masks = masks[:pred_boxes.size(0)]
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    # Show & save
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved visualization -> {out_path}")

if __name__ == "__main__":
    # 1) Read image -> bytes for endpoint
    img_bytes = load_image_bytes(IMAGE_PATH)

    # 2) Call endpoint
    pred = invoke_endpoint(img_bytes)
    # Expected base keys:
    #   "boxes": [[x1,y1,x2,y2], ...]
    #   "scores": [..]
    #   "labels": [..]
    # Optional for masks visual:
    #   "masks": [[[H x W floats 0..1] per instance], ...]  OR
    #   "masks_b64": ["<base64 PNG>", ...]

    # 3) Visualize
    visualize(IMAGE_PATH, pred, SCORE_THRESHOLD, MASK_THRESHOLD, OUT_IMAGE)
