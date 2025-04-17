import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import torchfile
import scipy.io

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

class LandmarkDataset(Dataset):
    def __init__(self, img_dir, t7_dir=None, mat_dir=None, transform=None):
        self.img_dir = img_dir
        self.t7_dir = t7_dir
        self.mat_dir = mat_dir
        self.transform = transform

        self.image_files = []

        for f in os.listdir(img_dir):
            if f.endswith('.jpg'):
                img_path = os.path.join(img_dir, f)

                if t7_dir and os.path.exists(os.path.join(t7_dir, f.replace(".jpg", ".t7"))):
                    self.image_files.append(f)
                elif mat_dir and os.path.exists(os.path.join(mat_dir, f.replace(".jpg", ".mat"))):
                    self.image_files.append(f)
                # else skip if no landmark file

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.t7_dir and os.path.exists(os.path.join(self.t7_dir, img_name.replace('.jpg', '.t7'))):
            landmark_path = os.path.join(self.t7_dir, img_name.replace('.jpg', '.t7'))
            data = torchfile.load(landmark_path)
            landmarks = np.array(data) * 256.0  # t7 normalized
        elif self.mat_dir and os.path.exists(os.path.join(self.mat_dir, img_name.replace('.jpg', '.mat'))):
            landmark_path = os.path.join(self.mat_dir, img_name.replace('.jpg', '.mat'))
            data = scipy.io.loadmat(landmark_path)
            pts = data['pt2d']  # adjust if your .mat uses another key
            landmarks = pts[:2].T  # shape (68, 2)
        else:
            raise FileNotFoundError(f"No landmark file found for image: {img_name}")

        
        landmarks = landmarks.reshape(68, 2)
        landmarks = landmarks.astype(np.float32)
        # Normalize to [0,1] relative to 256x256 image
        # landmarks = landmarks / 256.0
        landmarks = torch.tensor(landmarks, dtype=torch.float32).view(-1)

        return image, landmarks
