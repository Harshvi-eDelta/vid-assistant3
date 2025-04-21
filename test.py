import os
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import LandmarkCNN
from data import get_transforms,LandmarkDataset
from torch.utils.data import DataLoader
import cv2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Directories
DATA_DIR = '/Users/edelta076/Desktop/Project_VID_Assistant3/face_images'  # Path to the folder containing your images
MODEL_PATH = 'best_model.pth'  # Path to the trained model

# Load model
model = LandmarkCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))  # Load the best model

# Get transforms (for preprocessing)
transform = get_transforms()

# Prepare the dataset (we assume .mat and .jpg files are in the same directory)
dataset = LandmarkDataset(data_dir=DATA_DIR, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Inference
model.eval()  # Set model to evaluation mode

# Process all files in the directory
for img_file in sorted(os.listdir(DATA_DIR)):
    if img_file.endswith('.jpg'):
        # Get the image path and load the image
        img_path = os.path.join(DATA_DIR, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load {img_file}")
            continue
        
        # Apply the transformations (resize, normalize, etc.)
        img = transform(img)
        img = img.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
        
        # Get the landmarks prediction
        with torch.no_grad():  # Disable gradient computation for inference
            predicted_landmarks = model(img)
        
        # Convert predictions back to numpy and rescale to original image size
        predicted_landmarks = predicted_landmarks.cpu().numpy().flatten() * 256  # Rescale to image size
        
        # Print the predicted landmarks
        print(f"Predicted landmarks for {img_file}:")
        print(predicted_landmarks)
        
        # You can also visualize or save the results if needed
        # For example, drawing landmarks on the image (optional)
        img = cv2.imread(img_path)  # Reload the original image
        for i in range(0, len(predicted_landmarks), 2):  # Loop through landmarks (x, y pairs)
            x, y = int(predicted_landmarks[i]), int(predicted_landmarks[i+1])
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Draw green circle at each landmark
            
        # Save the image with landmarks (optional)
        output_img_path = os.path.join(DATA_DIR, f"landmarks_{img_file}")
        cv2.imwrite(output_img_path, img)  # Save the image with landmarks drawn 