'''import os
import torch
import numpy as np
import torchfile  # This is key for .t7 files
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LandmarkDataset(Dataset):
    def __init__(self, img_dir, t7_dir, transform=None):
        self.img_dir = img_dir
        self.t7_dir = t7_dir
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.img_dir, img_name)
        t7_path = os.path.join(self.t7_dir, os.path.splitext(img_name)[0] + '.t7')

        # Load image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # Load .t7 landmark
        landmark = torchfile.load(t7_path)  # (68, 2)
        landmark = np.array(landmark).astype(np.float32)

        # Normalize landmarks by original image size
        landmark[:, 0] /= original_width   # normalize x
        landmark[:, 1] /= original_height  # normalize y

        # Transform image
        if self.transform:
            image = self.transform(image)

        landmark = torch.tensor(landmark, dtype=torch.float32).view(-1)  # flatten to (136,)
        return image, landmark

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])'''

import os
import torch
import numpy as np
import torchfile  # This is key for .t7 files
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import h5py

class LandmarkDataset(Dataset):
    def __init__(self, img_dir, t7_dir, is_google=False, transform=None):
        self.img_dir = img_dir
        self.t7_dir = t7_dir
        self.is_google = is_google
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.img_dir, img_name)
        
        if self.is_google:
            t7_file_name = f"{img_name.split('.')[0]}_landmarks_landmarks.t7"
        else:
            t7_file_name = os.path.splitext(img_name)[0] + '.t7'

        t7_path = os.path.join(self.t7_dir, t7_file_name)

        # print(f"Trying to load image: {image_path}")
        # print(f"Trying to load landmark file: {t7_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # Load landmarks
        if self.is_google:
            # ➔ load using h5py
            with h5py.File(t7_path, 'r') as f:
                landmark = np.array(f['landmarks'])  # your dataset inside
        else:
            # ➔ old .t7 loading
            landmark = torchfile.load(t7_path)
            landmark = np.array(landmark)

        landmark = landmark.astype(np.float32)

        # Normalize landmarks by original image size
        landmark[:, 0] /= original_width
        landmark[:, 1] /= original_height

        # Transform image
        if self.transform:
            image = self.transform(image)

        landmark = torch.tensor(landmark, dtype=torch.float32).view(-1)  # flatten to (136,)
        return image, landmark

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])



