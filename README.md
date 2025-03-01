# Road Boundary Detection

## Overview

This project focuses on detecting **unmarked road boundaries** using a **custom-trained YOLOv8 segmentation model**. The model is trained on **1,400 annotated images** to segment road boundaries, particularly for **rural Indian roads** where standard methods fail. The project uses **Roboflow for dataset preparation** and **YOLOv8-seg for training and inference**.

## Features

- **Custom Road Boundary Segmentation Model**
- **Trained with YOLOv8-seg**
- **Supports Image & Video Input for Inference**
- **Utilizes Roboflow for Custom Dataset Annotation & Preprocessing**
- **Inference Results Saved as Annotated Videos/Images**

## Dataset Preparation with Roboflow

### Step 1: Annotate Images

1. Create a **free Roboflow account** ([Roboflow.com](https://roboflow.com)).
2. Upload your dataset and choose **"Object Detection"** or **"Instance Segmentation"**.
3. Use the annotation tool to draw **polygon-based annotations** around road boundaries.
4. Once annotation is complete, generate a dataset version.

### Step 2: Export Dataset

1. Click **"Generate"** to create a dataset version.
2. Download dataset in **YOLOv8 format** (`.zip` file containing images and label files).
3. Extract the dataset into the `data/` directory.

## Installation

### **Dependencies**

Install the required dependencies using:

```bash
pip install ultralytics opencv-python roboflow torch torchvision
```

Ensure that you have **YOLOv8 installed**:

```bash
pip install ultralytics
```

## Model Training

Use `model.py` to train your custom YOLOv8 segmentation model.

```bash
python model.py
```

### **Configuration in `model.py`**

- Change dataset path to your downloaded **Roboflow dataset**.
- Modify **batch size, epochs, and image size** as needed.

## Running Inference

Once the model is trained, use `results.py` to run inference on images/videos:

```bash
python results.py --weights best.pt --source test_video.mp4
```

### **Options for `results.py`**

- `--weights`: Path to trained YOLOv8 model weights.
- `--source`: Input image or video file.
- `--output`: Directory to save annotated results.

## Example Output

After running inference, the model will output:

- **Segmented Road Boundaries** in an image/video.
- Annotated results saved in `output/`.

## Future Improvements

- **Train on Larger Datasets** for improved accuracy.
- **Implement Real-Time Inference** for edge devices.
- **Enhance Post-Processing** for better boundary refinement.

## Acknowledgments

- [Roboflow](https://roboflow.com) for dataset annotation & processing.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for segmentation models.
