# import torch
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# from PIL import Image

# from model import MultiStageCNN
# from data import get_transforms, generate_heatmaps

# # -------- SETTINGS --------
# IMAGE_PATH = "./dataset/original_jpg/472.jpg"    # your input image
# T7_PATH = "./dataset/t7/472.t7"         # matching .t7 landmark file
# IS_GOOGLE = False                          # set True if Google-style t7
# SAVE_DIR = "./outputs_single"
# CHECKPOINT_PATH = "best_heatmap_model_2.pth"

# # -------- SETUP --------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.makedirs(SAVE_DIR, exist_ok=True)

# # -------- LOAD MODEL --------
# model = MultiStageCNN(stages=5).to(DEVICE)
# model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
# model.eval()

# # -------- LOAD IMAGE --------
# original_img = cv2.imread(IMAGE_PATH)
# original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
# original_h, original_w = original_img.shape[:2]
# resized_img = cv2.resize(original_img, (256, 256))
# pil_img = Image.fromarray(original_img)

# transform = get_transforms()
# input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

# # -------- INFERENCE --------
# with torch.no_grad():
#     heatmaps = model(input_tensor)[-1].squeeze(0).cpu().numpy()  # shape: (68, 128, 128)

# # -------- GET LANDMARK COORDS --------
# coords = []
# for hmap in heatmaps:
#     y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
#     coords.append((x * 2, y * 2))  # upscale to match 256x256

# # -------- DRAW LANDMARKS --------
# landmark_img = resized_img.copy()
# for x, y in coords:
#     cv2.circle(landmark_img, (int(x), int(y)), 2, (0, 255, 0), -1)

# cv2.imwrite(os.path.join(SAVE_DIR, "pred_landmarks_472.jpg"), cv2.cvtColor(landmark_img, cv2.COLOR_RGB2BGR))

# # -------- COMBINED HEATMAP --------
# combined_heatmap = np.sum(heatmaps, axis=0)
# combined_heatmap = np.clip(combined_heatmap, 0, 1)
# plt.figure(figsize=(4, 4))
# plt.imshow(combined_heatmap, cmap='hot')
# plt.axis('off')
# plt.savefig(os.path.join(SAVE_DIR, "combined_heatmap_472.png"), bbox_inches='tight', pad_inches=0)
# plt.close()

# # -------- BLENDED OVERLAY --------
# def overlay_heatmap_on_image(image, heatmap):
#     heatmap_resized = cv2.resize(heatmap, (256, 256))
#     heatmap_norm = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-6)
#     heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
#     return overlay

# blend = overlay_heatmap_on_image(resized_img, combined_heatmap)
# cv2.imwrite(os.path.join(SAVE_DIR, "blended_overlay_472.png"), blend)

# print("✅ Saved:")
# print("- pred_landmarks.jpg")
# print("- combined_heatmap.png")
# print("- blended_overlay.png")

import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

import torchfile
import h5py
from torch.nn import MSELoss

from model import MultiStageCNN
from data import get_transforms, generate_heatmaps

# -------- SETTINGS --------
IMAGE_PATH = "./dataset/original_jpg/764.jpg"
T7_PATH = "./dataset/t7/764.t7"
IS_GOOGLE = False
SAVE_DIR = "./outputs_single"
CHECKPOINT_PATH = "best_heatmap_model_2.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# -------- LOAD MODEL --------
model = MultiStageCNN(stages=5).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# -------- LOAD IMAGE --------
original_img = cv2.imread(IMAGE_PATH)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
original_h, original_w = original_img.shape[:2]
resized_img = cv2.resize(original_img, (256, 256))
pil_img = Image.fromarray(original_img)

transform = get_transforms()
input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

# -------- PREDICT --------
with torch.no_grad():
    pred_heatmaps = model(input_tensor)[-1].squeeze(0).cpu().numpy()  # (68, 128, 128)

# -------- GET LANDMARK COORDS --------
coords = []
for hmap in pred_heatmaps:
    y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
    coords.append((x * 2, y * 2))  # upscale to 256x256

# -------- DRAW LANDMARKS --------
landmark_img = resized_img.copy()
for x, y in coords:
    cv2.circle(landmark_img, (int(x), int(y)), 2, (0, 255, 0), -1)
cv2.imwrite(os.path.join(SAVE_DIR, "pred_landmarks_764.jpg"), cv2.cvtColor(landmark_img, cv2.COLOR_RGB2BGR))

# -------- COMBINED HEATMAP --------
combined_heatmap = np.sum(pred_heatmaps, axis=0)
combined_heatmap = np.clip(combined_heatmap, 0, 1)
plt.figure(figsize=(4, 4))
plt.imshow(combined_heatmap, cmap='hot')
plt.axis('off')
plt.savefig(os.path.join(SAVE_DIR, "combined_heatmap_764.png"), bbox_inches='tight', pad_inches=0)
plt.close()

# -------- BLENDED OVERLAY --------
def overlay_heatmap_on_image(image, heatmap):
    heatmap_resized = cv2.resize(heatmap, (256, 256))
    heatmap_norm = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-6)
    heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    return overlay

blend = overlay_heatmap_on_image(resized_img, combined_heatmap)
cv2.imwrite(os.path.join(SAVE_DIR, "blended_overlay_764.png"), blend)

# -------- LOAD GT LANDMARKS & COMPUTE MSE --------
if IS_GOOGLE:
    with h5py.File(T7_PATH, 'r') as f:
        landmark = np.array(f['landmarks'])
else:
    landmark = torchfile.load(T7_PATH)
    landmark = np.array(landmark)

# Normalize GT landmarks
landmark = landmark.astype(np.float32)
landmark[:, 0] /= original_w
landmark[:, 1] /= original_h

# Generate GT heatmaps
gt_heatmaps = generate_heatmaps(landmark, height=128, width=128).numpy()  # (68, 128, 128)

# Compute MSE
mse_criterion = MSELoss()
mse_value = mse_criterion(torch.tensor(pred_heatmaps), torch.tensor(gt_heatmaps)).item()

# -------- DONE --------
print("✅ Saved:")
print("- pred_landmarks.jpg")
print("- combined_heatmap.png")
print("- blended_overlay.png")
print(f"\n📊 MSE for image2 image: {mse_value:.6f}")

