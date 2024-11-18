#%%
import os
import random
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn

# Paths
dataset_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/frames_by_label'
output_dir = './grad_cam'
model_path = './model_weights/resnet34_bs_80_lr_2.5e-05_epoch_20_wd_0.0017_wlf_False.pth'

# Parameters
num_samples = 100  # Number of images to select per label
use_cuda = torch.cuda.is_available()

# Define a simple dataset to load images
class EardrumDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model and apply weights
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path))

model.eval()
if use_cuda:
    model = model.cuda()

# Define target layer for Grad-CAM
target_layers = [model.layer4[-1]]

# Select 100 random images from each label
def get_image_paths(label_folder, num_samples):
    label_path = os.path.join(dataset_dir, label_folder)
    all_images = os.listdir(label_path)
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    return [os.path.join(label_path, img) for img in selected_images]

label_0_images = get_image_paths('label_0', num_samples)
label_1_images = get_image_paths('label_1', num_samples)

# Combine paths and labels
all_image_paths = label_0_images + label_1_images
all_labels = [0] * len(label_0_images) + [1] * len(label_1_images)

# Prepare the dataset and dataloader
dataset = EardrumDataset(all_image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

# Create output directories
os.makedirs(os.path.join(output_dir, 'label_0'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'label_1'), exist_ok=True)

# Apply Grad-CAM and save images
for (input_tensor, img_path), label in zip(dataloader, all_labels):
    input_tensor = input_tensor.cuda() if use_cuda else input_tensor
    img_path = img_path[0]  # Get the path string from the tuple

    # Load the original image for overlay
    original_image = np.array(Image.open(img_path).convert("RGB"))
    original_image = cv2.resize(original_image, (224, 224))
    rgb_img = original_image / 255.0  # Normalize for visualization

    # Define target for Grad-CAM
    targets = [ClassifierOutputTarget(1)]

    # Generate Grad-CAM mask
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Overlay Grad-CAM on the image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save the image
    save_path = os.path.join(output_dir, f'label_{label}', os.path.basename(img_path))
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print(f"Saved Grad-CAM visualization for {img_path} to {save_path}")

print("Grad-CAM visualizations generated and saved successfully.")

# %%
