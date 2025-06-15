from ultralytics import YOLO
from PIL import Image
import cv2 # We use OpenCV for displaying the image
from IPython.display import display, Image

def main():
    # --- 1. Load Your Custom-Trained Model ---
    # Make sure the path points to YOUR 'best.pt' file from the training run.
    model_path = 'runs/detect/yolov5s_hardhat_test/weights/best.pt'
    model = YOLO(model_path)

    # --- 2. Make a Prediction ---
    # Path to an image you want to test. Use one from the 'test/images' folder.
    image_to_predict = 'datasets/valid/images/hard_hat_workers851_png.rf.4c94c44694ca45a756d64bf3a13c6671.jpg'

    print(f"Running inference on: {image_to_predict}")
    
    # The model returns a list of result objects
    results = model(image_to_predict)

    # --- 3. Process and Display Results ---
    # The result object contains all the information about the detections.
    for result in results:
        # The 'plot()' method conveniently draws the bounding boxes on the image.
        # It returns a NumPy array in BGR format (standard for OpenCV).
        img_with_boxes = result.plot()

        # Display the image using OpenCV

       # cv2.imshow("YOLOv5 Detection", img_with_boxes)
        
        # Or save the image to a file
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, img_with_boxes)
        print(f"Result saved to {output_path}")
    # If you want to display the image in a Jupyter Notebook, you can use:
    # display(Image(filename=output_path))
    # Wait for a key press to close the image window
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()