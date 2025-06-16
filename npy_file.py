import torch
import numpy as np
import cv2
from model import MultiStageCNN
from torchvision import transforms
import os

# ---- Config ----
IMAGE_PATH = './face_images/fimg1.jpg'
MODEL_PATH = './best_heatmap_model_2.pth'
SAVE_PATH = './npy_files/fimg1_landmarks.npy'
IMAGE_SIZE = 256  # Input image resized to 256x256

# ---- Utility: Decode heatmaps to coordinates ----
def get_landmarks_from_heatmap(heatmaps, orig_size):
    N, C, H, W = heatmaps.shape  # (1, 68, 64, 64)
    heatmaps_reshaped = heatmaps.view(N, C, -1)
    coords = heatmaps_reshaped.argmax(dim=2)  # shape (N, 68)
    coords = torch.stack([coords % W, coords // W], dim=2).float()  # shape (N, 68, 2)

    # Scale from heatmap space (64x64) to original image size
    scale_x = orig_size[0] / W
    scale_y = orig_size[1] / H
    coords[:, :, 0] *= scale_x
    coords[:, :, 1] *= scale_y

    return coords[0].numpy()  # shape (68, 2)

# ---- Load image ----
image = cv2.imread(IMAGE_PATH)
orig_h, orig_w = image.shape[:2]
image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
input_tensor = transforms.ToTensor()(image_resized).unsqueeze(0)

# ---- Load model ----
model = MultiStageCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ---- Predict & Decode ----
with torch.no_grad():
    output = model(input_tensor)
    final_heatmap = output[-1]  # last stage output, shape (1, 68, 64, 64)
    landmarks = get_landmarks_from_heatmap(final_heatmap, (orig_w, orig_h))  # shape (68, 2)

# ---- Normalize landmarks to [-1, 1] ----
landmarks_normalized = (landmarks / [orig_w, orig_h]) * 2 - 1  # shape (68, 2)

# ---- Save to .npy ----
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
np.save(SAVE_PATH, landmarks_normalized)
print(f"[INFO] Saved normalized landmarks to {SAVE_PATH}")
