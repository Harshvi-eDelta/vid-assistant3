'''import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import LandmarkDataset, get_transforms
from model import LandmarkCNN
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/original_jpg_copy"
t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/t7"
save_path = "best_model.pth"

# Dataset and Loader
dataset = LandmarkDataset(img_dir, t7_dir, transform=get_transforms())
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model, Loss, Optimizer
model = LandmarkCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss = float("inf")

# epochs = 30
epochs = 45

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, landmarks in loop:
        images, landmarks = images.to(device), landmarks.to(device)

        outputs = model(images)
        loss = criterion(outputs, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print("Best model saved.")'''

'''import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from data import LandmarkDataset, get_transforms
from model import LandmarkCNN
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for your datasets
original_img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/original_jpg"
original_t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/t7"
google_img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_original_jpg"  # path for Google images
google_t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_t7"  # path for Google .t7 files

save_path = "best_model_3.pth"

# Initialize datasets for both the original and Google image datasets
original_dataset = LandmarkDataset(img_dir=original_img_dir, t7_dir=original_t7_dir, transform=get_transforms())
google_dataset = LandmarkDataset(img_dir=google_img_dir, t7_dir=google_t7_dir, is_google=True, transform=get_transforms())

print(f"Original dataset size: {len(original_dataset)}")
print(f"Google dataset size: {len(google_dataset)}")

# Combine both datasets into one
combined_dataset = ConcatDataset([original_dataset, google_dataset])

# Dataset and Loader
train_loader = DataLoader(combined_dataset, batch_size=16, shuffle=True)

# Model, Loss, Optimizer
model = LandmarkCNN().to(device)
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # best_model_2
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)    # best_model_3

best_loss = float("inf")

# epochs = 30 
epochs = 55

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, landmarks in loop:
        images, landmarks = images.to(device), landmarks.to(device)

        outputs = model(images)
        loss = criterion(outputs, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print("Best model saved.")'''

# heatmap - multiple stage 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from data import LandmarkDataset, get_transforms
from model import MultiStageCNN
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_dataset = LandmarkDataset("/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/original_jpg", 
                                   "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/t7", transform=get_transforms())
google_dataset = LandmarkDataset("/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_original_jpg", 
                                "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_t7", is_google=True, transform=get_transforms())
combined_dataset = ConcatDataset([original_dataset, google_dataset])

train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
model = MultiStageCNN(stages=5).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_loss = float("inf")

for epoch in range(50):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")

    for images, gt_heatmaps in loop:
        images, gt_heatmaps = images.to(device), gt_heatmaps.to(device)
        preds = model(images)  # List of 5 outputs

        loss = sum(criterion(pred, gt_heatmaps) for pred in preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_heatmap_model_2.pth")
        print("Saved Best Heatmap Model")

