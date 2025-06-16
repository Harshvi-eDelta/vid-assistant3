'''import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import LandmarkCNN
from data import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LandmarkCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Load custom image
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant3/face_images/fimg1.jpg"         # 1,3,4,5,7,8,10,11,14,18,19,20,041
original_img = cv2.imread(image_path)

if original_img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Convert BGR (OpenCV) to RGB
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Resize for visualization later
resized_img = cv2.resize(original_img, (256, 256))

# Convert to PIL for transform
pil_img = Image.fromarray(original_img)

# Apply transform
transform = get_transforms()
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# Predict landmarks
with torch.no_grad():
    output = model(input_tensor).cpu().numpy().reshape(-1, 2)
    
# Denormalize using the final display image size (256x256)
output[:, 0] *= 256  # x
output[:, 1] *= 256  # y

# Draw landmarks
for (x, y) in output:
    cv2.circle(resized_img, (int(x), int(y)), 2, (0, 255, 0), -1)

# Show image with landmarks
plt.figure(figsize=(4,4))
plt.imshow(resized_img)
plt.title("Predicted Landmarks")
plt.axis("off")
plt.show()'''

'''import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import LandmarkCNN
from data import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LandmarkCNN()
model.load_state_dict(torch.load("best_model_3.pth", map_location=device,weights_only=True))
model.to(device)
model.eval()

# Load custom image (Ensure this image exists in your directory)
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant3/face_images/image20.jpg"         # 3,4,5,7,8,12,13,14,15,18,19,041,048,060,23,24,27,28,29       
original_img = cv2.imread(image_path)                                                        # 1,14,_1,4,5,7,9,11,12,13,14,16,17,18,19,24,25,_1,041,044,0133,27,28
                                                                                             
if original_img is None:                                                                     
    raise FileNotFoundError(f"Image not found at path: {image_path}")    # 1,_1,1,4,5,11,12,13,14,15,18,19,27,28,29,30,3,4
                                                                                                
# Convert BGR (OpenCV) to RGB
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)    # _1,4,5,10,11,12,13,15,16,18,23,24,27,28,29,30,041,046,048                    

# Resize for visualization later
resized_img = cv2.resize(original_img, (256, 256))           

# Convert to PIL for transform
pil_img = Image.fromarray(original_img)

# Apply transform (resize to 256x256 and normalization)
transform = get_transforms()
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# Predict landmarks
with torch.no_grad():
    output = model(input_tensor).cpu().numpy().reshape(-1, 2)

# Denormalize using the final display image size (256x256)
output[:, 0] *= 256  # x
output[:, 1] *= 256  # y

print(output)

# Draw landmarks on the image
for (x, y) in output:
    cv2.circle(resized_img, (int(x), int(y)), 2, (0, 255, 0), -1)

# Show image with landmarks
plt.figure(figsize=(4,4))
plt.imshow(resized_img)
plt.savefig("abc")
plt.title("Predicted Landmarks")
plt.axis("off")
plt.show()'''


# heatmap - multiple stage

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

image_path = "./face_images/fimg57.jpg"    # _1,4,27,1,2,image15,i16,i17,i20
original_img = cv2.imread(image_path)                                               
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)            # 1,27
resized_img = cv2.resize(original_img, (256, 256))
pil_img = Image.fromarray(original_img)

input_tensor = get_transforms()(pil_img).unsqueeze(0).to(device)        # 1,27,32,35
with torch.no_grad():
    heatmaps = model(input_tensor)[-1].squeeze(0).cpu().numpy()  # Final stage

coords = []
for hmap in heatmaps:
    y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
    coords.append((x * 2, y * 2))  # Scale up from 64 to 256        # x * 4 and y * 4 causing and error !
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


