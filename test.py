import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import MultiStageCNN
from data import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiStageCNN(stages=5)
model.load_state_dict(torch.load("best_heatmap_model_2.pth", map_location=device))
model.to(device).eval()

image_path = "./face_images/fimg25.jpg"    # 56,52,49,48,10,2,4
original_img = cv2.imread(image_path)                                               
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)            
resized_img = cv2.resize(original_img, (256, 256))
pil_img = Image.fromarray(original_img)

input_tensor = get_transforms()(pil_img).unsqueeze(0).to(device)        
with torch.no_grad():
    heatmaps = model(input_tensor)[-1].squeeze(0).cpu().numpy() 

coords = []
for hmap in heatmaps:
    y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
    coords.append((x * 2, y * 2))  
print()
print(len(coords))
print(coords)
print()

for x, y in coords:
    cv2.circle(resized_img, (int(x), int(y)), 2, (0, 255, 0), -1)

plt.figure(figsize = (4,4))
plt.imshow(resized_img)
plt.title("Predicted Landmarks")
plt.axis("off")
plt.show()


