# Step 1: Install Required Libraries
!pip install ultralytics  # Install Ultralytics library for YOLOv8
!pip install matplotlib   # For plotting graphs

from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Step 3: Define Dataset Paths
# Replace these paths with the actual paths to your dataset
train_path = "/content/Road-Boundary-Detection/train"
val_path = "/content/Road-Boundary-Detection/valid"
test_path = "/content/Road-Boundary-Detection/test"

# Step 4: Create a YAML Configuration File for the Dataset
yaml_content = f"""
train: {train_path}/images
val: {val_path}/images
test: {test_path}/images

nc: 1  # Number of classes
names: ['road_boundary']  # Class names
"""

with open("road_boundary_segmentation_dataset.yaml", "w") as file:
    file.write(yaml_content)

# Step 5: Load a Pretrained YOLOv8 Segmentation Model (Medium Size)
model = YOLO("yolov8m-seg.pt")  # n,m,l,x depending on your requirement size, better model better accuracy

# Step 6: Train the Model
results = model.train(
    data="road_boundary_segmentation_dataset.yaml",
    epochs=25,  #depends on your vram 
    imgsz=640,  #depends on your annotations, take a multiple of 32
    batch=16,
    name="road_boundary_segmentation"
)

# Step 7: Plot Training Metrics
# ------------------------------------------
# Extract metrics from training results
metrics = results.results  # DataFrame containing all metrics

# Create subplots
plt.figure(figsize=(15, 10))

# Plot Training Losses
plt.subplot(2, 2, 1)
plt.plot(metrics['train/box_loss'], label='Box Loss')
plt.plot(metrics['train/seg_loss'], label='Segmentation Loss')
plt.plot(metrics['train/cls_loss'], label='Classification Loss')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Losses
plt.subplot(2, 2, 2)
plt.plot(metrics['val/box_loss'], label='Box Loss')
plt.plot(metrics['val/seg_loss'], label='Segmentation Loss')
plt.plot(metrics['val/cls_loss'], label='Classification Loss')
plt.title('Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot mAP Metrics
plt.subplot(2, 2, 3)
plt.plot(metrics['metrics/mAP50(B)'], label='mAP@50')
plt.plot(metrics['metrics/mAP50-95(B)'], label='mAP@50-95')
plt.title('mAP Metrics')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()

# Plot Learning Rate
plt.subplot(2, 2, 4)
plt.plot(metrics['lr/pg0'], label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")  # Save the plots
plt.show()

# Step 8: Final Output Message
print("Training completed successfully!")
print(f"Model weights saved to: {model.trainer.save_dir}")
print("Training metrics plots saved as 'training_metrics.png'")
