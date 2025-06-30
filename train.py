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
        preds = model(images) 

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

