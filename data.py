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

'''import os
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
        self.transform = transform

        self.image_files = []
        
        # Filter images that have corresponding t7
        for f in sorted(os.listdir(img_dir)):
            if f.endswith(('.jpg', '.png')):
                if self.is_google:
                    t7_file_name = f"{os.path.splitext(f)[0]}_landmarks_landmarks.t7"
                else:
                    t7_file_name = os.path.splitext(f)[0] + '.t7'

                t7_path = os.path.join(t7_dir, t7_file_name)
                
                if os.path.exists(t7_path):
                    self.image_files.append(f)
                else:
                    print(f"Skipping {f} because {t7_file_name} not found.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.img_dir, img_name)
        
        if self.is_google:
            t7_file_name = f"{os.path.splitext(img_name)[0]}_landmarks_landmarks.t7"
        else:
            t7_file_name = os.path.splitext(img_name)[0] + '.t7'

        t7_path = os.path.join(self.t7_dir, t7_file_name)

        # Load image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # Load landmarks
        if self.is_google:
            with h5py.File(t7_path, 'r') as f:
                landmark = np.array(f['landmarks'])
        else:
            landmark = torchfile.load(t7_path)
            landmark = np.array(landmark)

        landmark = landmark.astype(np.float32)

        # Normalize landmarks
        landmark[:, 0] /= original_width
        landmark[:, 1] /= original_height

        # Transform image
        if self.transform:
            image = self.transform(image)

        landmark = torch.tensor(landmark, dtype=torch.float32).view(-1)  # flatten (68x2) -> (136,)
        return image, landmark


def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

# def get_transforms(train=True):
#     if train:
#         return transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
#             transforms.RandomRotation(degrees=10),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
#         ])
#     else:
#         return transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
#         ])
'''

# heatmap - multiple stage 

import os
import torch
import numpy as np
import torchfile
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import h5py
import cv2

def generate_heatmaps(landmarks, height, width, sigma=1):
    num_landmarks = landmarks.shape[0]
    heatmaps = np.zeros((num_landmarks, height, width), dtype=np.float32)

    for i, (x, y) in enumerate(landmarks):
        if x < 0 or y < 0 or x >= 1 or y >= 1:
            continue
        x = int(x * width)
        y = int(y * height)

        heatmap = np.zeros((height, width), dtype=np.float32)
        tmp_size = sigma * 3
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]

        if ul[0] >= width or ul[1] >= height or br[0] < 0 or br[1] < 0:
            continue

        size = 2 * tmp_size + 1
        x_range = np.arange(0, size, 1, float)
        y_range = x_range[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))

        g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
        img_x = max(0, ul[0]), min(br[0], width)
        img_y = max(0, ul[1]), min(br[1], height)

        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        heatmaps[i] = heatmap

    return torch.tensor(heatmaps)

class LandmarkDataset(Dataset):
    def __init__(self, img_dir, t7_dir, is_google=False, transform=None):
        self.img_dir = img_dir
        self.t7_dir = t7_dir
        self.is_google = is_google
        self.transform = transform
        self.image_files = []

        for f in sorted(os.listdir(img_dir)):
            if f.endswith(('.jpg', '.png')):
                if self.is_google:
                    t7_file_name = f"{os.path.splitext(f)[0]}_landmarks_landmarks.t7"
                else:
                    t7_file_name = os.path.splitext(f)[0] + '.t7'

                t7_path = os.path.join(t7_dir, t7_file_name)
                if os.path.exists(t7_path):
                    self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.img_dir, img_name)

        if self.is_google:
            t7_file_name = f"{os.path.splitext(img_name)[0]}_landmarks_landmarks.t7"
        else:
            t7_file_name = os.path.splitext(img_name)[0] + '.t7'
        t7_path = os.path.join(self.t7_dir, t7_file_name)

        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        if self.is_google:
            with h5py.File(t7_path, 'r') as f:
                landmark = np.array(f['landmarks'])
        else:
            landmark = torchfile.load(t7_path)
            landmark = np.array(landmark)

        landmark = landmark.astype(np.float32)
        landmark[:, 0] /= original_width
        landmark[:, 1] /= original_height

        if self.transform:
            image = self.transform(image)

        heatmaps = generate_heatmaps(landmark, 64, 64)  # Generate heatmaps
        return image, heatmaps

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to 256x256
        transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])




