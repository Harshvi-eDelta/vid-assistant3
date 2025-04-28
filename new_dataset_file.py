import os
import cv2
import dlib
import torch

# Define paths
'''input_image_folder = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_original_jpg"  # Folder containing the Google Images
output_t7_folder = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_pt"    # Folder where the .t7 files will be saved

# Make sure the output folder exists
os.makedirs(output_t7_folder, exist_ok=True)

# Load Dlib's pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from Dlib

def get_landmarks(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = detector(gray)
    
    if len(faces) > 0:
        # Get the landmarks for the first detected face
        landmarks = predictor(gray, faces[0])
        
        # Return the list of (x, y) coordinates of the 68 landmarks
        return [(p.x, p.y) for p in landmarks.parts()]
    
    return None

def save_landmarks_as_pt(landmarks, pt_path):
    # Convert the landmarks into a tensor and save it as a .pt file
    landmarks_tensor = torch.tensor(landmarks).float()
    torch.save(landmarks_tensor, pt_path)

# Process each image in the input folder
for image_name in os.listdir(input_image_folder):
    if image_name.endswith(".jpg") or image_name.endswith(".jpeg") or image_name.endswith(".png"):
        # Full image path
        image_path = os.path.join(input_image_folder, image_name)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Get the landmarks for the image
        landmarks = get_landmarks(image)
        
        if landmarks:
            # Match the name convention of your older dataset for .pt file
            # Example: image_name = "google_image1.jpg", then the .pt file will be "google_image1_landmarks.pt"
            pt_file_name = image_name.split('.')[0] + '_landmarks.pt'
            pt_path = os.path.join(output_t7_folder, pt_file_name)
            
            # Save the landmarks as a .pt file
            save_landmarks_as_pt(landmarks, pt_path)
            print(f"Saved landmarks for {image_name} to {pt_file_name}")
        else:
            print(f"No face detected in {image_name}") '''

'''import os
import torch
import h5py  # You need to install h5py (pip install h5py)

# Define paths
input_pt_folder = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_pt"  # Folder containing the .pt files
output_t7_folder = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_t7"  # Folder where the .t7 files will be saved

# Make sure the output folder exists
os.makedirs(output_t7_folder, exist_ok=True)

def convert_pt_to_t7(pt_file_path, t7_file_path):
    # Load the PyTorch tensor from the .pt file
    landmarks_tensor = torch.load(pt_file_path)
    
    # Create the .t7 file using h5py
    with h5py.File(t7_file_path, 'w') as f:
        # Save the tensor as a dataset in the HDF5 file
        f.create_dataset('landmarks', data=landmarks_tensor.numpy())
    print(f"Converted {pt_file_path} to {t7_file_path}")

# Process each .pt file in the input folder
for pt_file_name in os.listdir(input_pt_folder):
    if pt_file_name.endswith(".pt"):
        # Full file paths
        pt_file_path = os.path.join(input_pt_folder, pt_file_name)
        t7_file_name = pt_file_name.replace(".pt", "_landmarks.t7")  # Match the name convention for .t7 files
        t7_file_path = os.path.join(output_t7_folder, t7_file_name)
        
        # Convert the .pt file to .t7
        convert_pt_to_t7(pt_file_path, t7_file_path)'''

# to view generated landmarks on google images

import os
import cv2
import h5py
import matplotlib.pyplot as plt

# Define paths
input_image_folder = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_original_jpg"  # Folder containing the Google Images
input_t7_folder = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_t7"  # Folder where the .t7 files are saved

def load_landmarks_from_t7(t7_file_path):
    # Load the .t7 file using h5py
    with h5py.File(t7_file_path, 'r') as f:
        # Access the landmarks dataset
        landmarks = f['landmarks'][:]
    return landmarks

def visualize_landmarks(image_path, landmarks):
    # Debugging: Print the image path to verify it exists
    print(f"Loading image from: {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the path.")
        return
    
    # Convert from BGR to RGB for proper display in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.imshow(image_rgb)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=10)  # Overlay landmarks as red dots
    plt.title(f"Landmarks on {os.path.basename(image_path)}")
    plt.axis('off')  # Hide axes
    plt.show()

# Process each .t7 file in the input folder
for t7_file_name in os.listdir(input_t7_folder):
    if t7_file_name.endswith(".t7"):
        # Full file paths
        t7_file_path = os.path.join(input_t7_folder, t7_file_name)
        image_file_name = t7_file_name.replace("_landmarks_landmarks.t7", ".jpg")  # Assuming the same name convention for images
        image_path = os.path.join(input_image_folder, image_file_name)
        
        # Visualize the image with landmarks
        landmarks = load_landmarks_from_t7(t7_file_path)
        visualize_landmarks(image_path, landmarks)


'''import h5py

# Define the path to the .t7 file
t7_file_path = "/Users/edelta076/Desktop/Project_VID_Assistant3/dataset/new_t7/n1_landmarks_landmarks.t7"

# Load the .t7 file using h5py
with h5py.File(t7_file_path, 'r') as f:
    # Access the dataset stored in the .t7 file (which contains the landmarks)
    landmarks = f['landmarks'][:]

print(landmarks)  # Print the loaded landmarks'''





