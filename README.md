# Vision Models

A computer vision project using PyTorch. Sagemaker train, deploy endpoint, edge infer. 

## 🚀 Features

- **Mask R-CNN Implementation**: Instance segmentation for pedestrian detection using Penn-Fudan dataset
- **YOLOv5 Implementation**: Object detection for hard hat detection in construction safety
- **SageMaker Integration**: Ready for AWS SageMaker deployment
- **GPU Support**: Optimized for CUDA-enabled GPUs
- **Comprehensive Training Pipeline**: Complete training, validation, and inference workflows

## 📁 Project Structure

```
vision_models/
├── rcnn/                          # Mask R-CNN implementation
│   ├── data/
│   │   └── PennFudanPed/         # Pedestrian detection dataset
│   ├── model/                     # Trained model weights
│   ├── train.py                   # Training script
│   ├── infer.py                   # Inference script
│   ├── deploy_endpoint.py         # SageMaker deployment
│   ├── requirements.txt           # Dependencies
│   └── utils.py                   # Utility functions
├── YOLO/                          # YOLOv5 implementation
│   ├── datasets/                  # Hard hat detection dataset
│   ├── runs/                      # Training outputs
│   ├── train.py                   # YOLO training script
│   ├── inference.py               # YOLO inference script
│   └── hardHat_download.ipynb     # Dataset preparation
├── requirements.txt               # Main project dependencies
└── venv/                         # Virtual environment
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vision_models
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Mask R-CNN (Pedestrian Detection)

#### Training

1. **Prepare the dataset**
   - The Penn-Fudan dataset is already included in `rcnn/data/PennFudanPed/`
   - Dataset structure:
     ```
     PennFudanPed/
     ├── PNGImages/          # Input images
     ├── PedMasks/           # Segmentation masks
     └── Annotation/         # Bounding box annotations
     ```

2. **Train the model**
   ```bash
   cd rcnn
   python train.py
   ```

   The training script will:
   - Load the Mask R-CNN model pre-trained on COCO
   - Fine-tune on the Penn-Fudan dataset
   - Save the trained model to `model/model.pth`

#### Inference

```bash
cd rcnn
python infer.py
```

This will:
- Load the trained model
- Run inference on a sample image
- Display results with bounding boxes and segmentation masks

#### SageMaker Deployment

```bash
cd rcnn
python deploy_endpoint.py
```

### YOLOv5 (Hard Hat Detection)

#### Training

1. **Dataset Preparation**
   - The hard hat dataset is located in `YOLO/datasets/`
   - Dataset configuration is in `YOLO/datasets/data.yaml`

2. **Train the model**
   ```bash
   cd YOLO
   python train.py
   ```

   Training parameters:
   - **Epochs**: 10
   - **Image size**: 640x640
   - **Batch size**: 8
   - **Model**: YOLOv5s (small variant)

#### Inference

```bash
cd YOLO
python inference.py
```

This will:
- Load the trained YOLOv5 model
- Run inference on a validation image
- Save the result as `detection_result.jpg`

## 📊 Models

### Mask R-CNN
- **Architecture**: Mask R-CNN with ResNet-50-FPN backbone
- **Task**: Instance segmentation
- **Classes**: Background + Pedestrian
- **Dataset**: Penn-Fudan Pedestrian Dataset
- **Performance**: COCO-style evaluation metrics

### YOLOv5
- **Architecture**: YOLOv5s (small variant)
- **Task**: Object detection
- **Classes**: Hard hat detection
- **Dataset**: Custom hard hat dataset
- **Performance**: mAP, precision, recall metrics

## 🔧 Configuration

### Training Parameters

#### Mask R-CNN
- **Learning rate**: 0.005
- **Momentum**: 0.9
- **Weight decay**: 0.0005
- **Batch size**: 2
- **Epochs**: Configurable (default: 1 for quick testing)

#### YOLOv5
- **Learning rate**: Auto-scaled
- **Batch size**: 8
- **Epochs**: 10
- **Image size**: 640x640

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 8GB+ VRAM
- **Optimal**: NVIDIA RTX 3080 or better

## 🚀 Deployment

### Local Deployment

Both models can be run locally for inference:

```bash
# Mask R-CNN
cd rcnn
python infer.py

# YOLOv5
cd YOLO
python inference.py
```

### AWS SageMaker Deployment

The R-CNN implementation includes SageMaker deployment scripts:

1. **Model Packaging**
   ```bash
   cd rcnn
   python deploy_endpoint.py
   ```

2. **Endpoint Configuration**
   - Instance type: `ml.m5.large` (CPU) or `ml.g4dn.xlarge` (GPU)
   - Framework: PyTorch 1.13
   - Python version: 3.8
