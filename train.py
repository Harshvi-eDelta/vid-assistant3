import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from data import LandmarkDataset, get_transforms
from model import LandmarkCNN
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
t7_img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/original_jpg"
t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/t7"

mat_img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/HELEN"
mat_dir = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/HELEN"

save_path = "best_model.pth"

# Datasets
transform = get_transforms()
t7_dataset = LandmarkDataset(t7_img_dir, t7_dir=t7_dir, transform=transform)
mat_dataset = LandmarkDataset(mat_img_dir, mat_dir=mat_dir, transform=transform)

# Combine both datasets
full_dataset = ConcatDataset([t7_dataset, mat_dataset])
train_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)

# Model, Loss, Optimizer
model = LandmarkCNN().to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss = float("inf")
epochs = 30

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, landmarks in loop:
        images, landmarks = images.to(device), landmarks.to(device)

        # Ensure landmarks are of the shape (batch_size, 136)
        landmarks = landmarks.view(-1, 136)
        # Normalize landmarks to the [0, 1] range relative to a 256x256 image size
        landmarks = landmarks / 256.0


        outputs = model(images)
        loss = criterion(outputs, landmarks)

        # Print predicted vs ground truth landmarks (only once)
        if epoch == 0 and loop.n == 0:
            pred_landmarks = outputs[0].view(-1, 2).cpu().detach().numpy() * 256.0
            gt_landmarks = landmarks[0].view(-1, 2).cpu().numpy()

            print("Predicted landmarks (first 5):", pred_landmarks[:5].flatten())
            print("Ground truth landmarks (first 5):", gt_landmarks[:5].flatten())

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
        print(" Best model saved.")
