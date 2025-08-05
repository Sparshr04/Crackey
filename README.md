YOLOv12 Crack Detection
=======================

This repository contains the code and resources for training and inferencing a YOLOv12 model for crack detection, specifically utilizing a dataset obtained from Roboflow.

Table of Contents
-----------------

1.  [Project Description](#1-project-description)
2.  [Dataset](#2-dataset)
3.  [Prerequisites](#3-prerequisites)
4.  [Installation](#4-installation)
5.  [Training the Model](#5-training-the-model)
6.  [Inference (Making Predictions)](#6-inference-making-predictions)
    

1\. Project Description
-----------------------

This project aims to develop a robust crack detection system using the YOLO (You Only Look Once) object detection framework. The model is trained on a specialized dataset of images containing various types of cracks, sourced and preprocessed using Roboflow. The goal is to accurately identify and localize cracks in images, which can be crucial for structural health monitoring, quality control, and maintenance.

**Note on YOLOv12:** This project assumes a "YOLOv12" implementation. The commands and structure provided here are generalized and should be adaptable. If "YOLOv12" refers to a custom fork or an experimental version, minor adjustments to commands (especially train.py and detect.py arguments) might be necessary based on your specific framework. For Ultralytics YOLO, the commands are typically very similar across versions.

2\. Dataset
-----------

The model is trained on a crack detection dataset.

**Dataset Source:** Roboflow
**Dataset Name:** [Concrete cracks](https://universe.roboflow.com/susu-5j4cz/concrete-cracks-pvvid)
**Dataset Format:** YOLOv12 (You can choose a different format if you wish)

**To download your dataset from Roboflow:**

1.  Go to your Roboflow project page.
    
2.  Select the desired version of your dataset.
    
3.  Choose the export format (e.g., "YOLOv12 in our case).
    
4.  Copy the provided download code snippet.
    
5.  Place the downloaded dataset in the data/ directory (or wherever your data.yaml points).
    


3\. Prerequisites
-----------------

Before you begin, ensure you have the following installed:

*   Python 3.9+
    
*   pip (Python package installer)
    
*   (Optional but Recommended) A virtual environment (e.g., venv or conda)
    
*   NVIDIA GPU with CUDA (for GPU acceleration, highly recommended for training)
    

4\. Installation
----------------

1.  git clone https://github.com/Sparshr04/Crackey.git
    
2.  #### On Windows
    ```bash
    python -m venv venv
    \venv\Scripts\activate
    ``` 
    #### On macOS
    ```bash
    python -m venv venv
    source venv/bin/activate
    ``` 
  
3.  #### Requirements
    ```bash
    pip install -r requirements.txt 
    #(you might also directly use pip install ultralytics)
    ```
    

5\. Training the Model
----------------------

Training involves configuring your model, dataset, and training parameters.

1.  **Prepare your dataset:** Ensure your Roboflow dataset is downloaded and placed in the correct directory, and your data.yaml file is configured to point to the train, val, and test image directories and defines your classes.
    
2.  **Choose a pre-trained model (optional but recommended):** You can start training from a pre-trained YOLOv12 model checkpoint to leverage transfer learning. Common options include yolov12n.pt (nano), yolov12s.pt (small), etc. Place this .pt file in your project root or a designated weights/ directory.
    
3.  yolo train model=yolov12n.pt data=data.yaml epochs=100 imgsz=640 batch=16 **Explanation of arguments:Monitoring Training:** Training progress, metrics (mAP, loss), and visualizations will be saved in the runs/train/your\_run\_name/ directory.
    
    *   yolo train: The command to start training (for Ultralytics YOLO).
        
    *   model=yolov12.pt: Path to your initial model weights (e.g., a pre-trained model or a blank model). If yolov12.pt is not a standard Ultralytics model, replace it with yolov8n.pt or similar.
        
    *   data=data.yaml: Path to your dataset configuration file.
        
    *   epochs=100: Number of training epochs. Adjust based on your dataset size and convergence.
        
    *   imgsz=640: Input image size (e.g., 640x640 pixels).
        
    *   batch=16: Batch size. Adjust based on your GPU memory.
        

6\. Inference
----------------------------------

Once your model is trained, you can use it to detect cracks in new images or videos.

1.  **Locate your trained weights:** After training, your best model weights will be saved in runs/train/your\_run\_name/weights/best.pt.

2. CLI Interface is easy to use. Below are the usge examples for cli.


#### For a single image:
```bash
yolo detect predict model=runs/train/yolov12_crack_detection/weights/best.pt source=path/to/your/image.jpg
```

#### For a directory of images:
```bash
yolo detect predict model=runs/train/yolov12_crack_detection/weights/best.pt source=path/to/your/image_folder/
```

#### For a video file:
```bash
yolo detect predict model=runs/train/yolov12_crack_detection/weights/best.pt source=path/to/your/video.mp4
```

#### For webcam inference:
```bash
yolo detect predict model=runs/train/yolov12_crack_detection/weights/best.pt source=0
```
*Note: '0' represents the default webcam*

### Explanation of Arguments

- **model**: Path to the trained YOLOv12 crack detection model weights
- **source**: Input source (image file, directory, video file, or webcam index)

## Inference Results

The images/videos with detected cracks will be saved in a new directory, typically `runs/detect/predict/`.
        
