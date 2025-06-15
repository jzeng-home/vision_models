import torch
from ultralytics import YOLO

def main():
    # --- 1. PyTorch Device Setup ---
    # Check if a CUDA-enabled GPU is available, otherwise use the CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 2. Load a Pre-trained Model ---
    # We use 'yolov5s.pt'. 's' stands for "small". 
    # This model was pre-trained on the large COCO dataset.
    # The library automatically downloads the model weights for you.
    model = YOLO('yolov5s.pt')
    model.to(device) # Move the model to the selected device

    # --- 3. Train the Model (Transfer Learning) ---
    # The '.train()' method is the heart of the process.
    # It encapsulates the entire PyTorch training loop:
    #   - Data loading (using PyTorch DataLoaders internally)
    #   - Forward pass (getting predictions)
    #   - Loss calculation (comparing predictions to labels)
    #   - Backward pass (calculating gradients)
    #   - Optimizer step (updating model weights)
    
    print("Starting training...")
    # The 'data' argument points to our 'data.yaml' file.
    # This file tells YOLO where the training and validation images are.
    results = model.train(
        data='datasets/data.yaml',
        epochs=10,          # Number of training epochs. Keep it small for a quick tutorial.
        imgsz=640,          # Image size for training.
        batch=8,            # Batch size. Reduce if you run out of GPU memory.
        name='yolov5s_hardhat_test' # A name for the output folder
    )
    print("Training finished.")
    print("Results saved to:", results.save_dir)

if __name__ == '__main__':
    main()