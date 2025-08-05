import os
from ultralytics import YOLO

def main():
    """
    Main function to run the YOLOv12 training process for crack detection.
    """
    print("Starting YOLOv12 training process for crack detection...")

    # --- 1. Configuration ---
    # Define the path to your dataset's YAML file.
    # IMPORTANT: Make sure the paths inside your data.yaml file are correct!
    data_yaml_path = '/path/to/your/dataset/data.yaml'

    # Choose a pre-trained YOLOv12 model to start from.
    # 'yolov12n.pt' is the smallest and fastest, great for starting.
    # Other options: 'yolov12s.pt', 'yolov12m.pt', 'yolov12l.pt', 'yolov12x.pt'
    model_variant = 'yolov12.pt'

    # Training parameters
    training_epochs = 50  # Number of times to loop through the entire dataset
    image_size = 640      # Resize all images to this size for training

    try:
        print(f"Loading pre-trained model: {model_variant}")
        model = YOLO(model_variant)
        print("Training initialized. This may take a while depending on your hardware and dataset size.")
        print(f"Dataset YAML: {data_yaml_path}")
        print(f"Number of epochs: {training_epochs}")
        print(f"Image size: {image_size}")

        results = model.train(
            data=data_yaml_path,
            epochs=training_epochs,
            imgsz=image_size,
            project='runs/train',
            name='yolov12_crack_detection',
            exist_ok=True
        )

        print("\nTraining finished successfully!")
        print("Results, weights, and logs are saved in the 'runs/train/yolov8_crack_detection' directory.")
        print("The best model weights are saved as 'best.pt'. You can use this file for inference.")

    except FileNotFoundError:
        print(f"Error: The data YAML file was not found at '{data_yaml_path}'.")
        print("Please ensure the path is correct and the file exists.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the training process: {e}")
        print("Please check your ultralytics installation, dataset paths, and file permissions.")


if __name__ == '__main__':
    main()

