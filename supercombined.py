import torch
import cv2
import numpy as np
import PIL
from PIL import Image
import requests
import matplotlib.pyplot as plt
print(PIL.__version__)
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from SuperGluePretrainedNetwork.models.matching import Matching  # from Magic Leap's SuperGlue repo

# def load_image(url):
#     img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     return img

# image1 = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
# image2 = load_image("http://images.cocodataset.org/test-stuff2017/000000000568.jpg")
# images = [image1, image2]

def load_image(url):
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return img

# Use the provided image URLs
url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"

img1_path = "image_0/000000.png"
img2_path = "image_0/000005.png"


image1 = Image.open(img1_path).convert("RGB")  # or "L" for grayscale
#image1 = np.array(image1)

image2 = Image.open(img2_path).convert("RGB")  # or "L" for grayscale
#image2 = np.array(image2)

# image1 = load_image(url_image1)
# image2 = load_image(url_image2)

images = [image1, image2]

# Load SuperPoint from HuggingFace
print("Loading SuperPoint model...")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

# Process images through SuperPoint
inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

sizes = [(img.height, img.width) for img in images]
features = processor.post_process_keypoint_detection(outputs, sizes)

# Prepare SuperGlue input
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Loading SuperGlue model (outdoor weights)...")
matching = Matching({
    'superpoint': {},
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}).eval().to(device)

# Format SuperPoint features
def format_features(feat):
    return {
        'keypoints': feat["keypoints"][None].to(device),          # [1, N, 2]
        'descriptors': feat["descriptors"][None].permute(0, 2, 1).to(device),  # [1, D, N]
        'scores': feat["scores"][None].to(device)                 # [1, N]
    }

data0 = format_features(features[0])
data1 = format_features(features[1])

# Get image shape for normalization
image_shape = (image1.height, image1.width)  # (H, W)

# Prepare full input dict for SuperGlue
data = {
    'keypoints0': data0['keypoints'],
    'keypoints1': data1['keypoints'],
    'descriptors0': data0['descriptors'],
    'descriptors1': data1['descriptors'],
    'scores0': data0['scores'],
    'scores1': data1['scores'],
    'image0': torch.empty(1, 1, *image_shape).to(device),  # dummy [1, 1, H, W]
    'image1': torch.empty(1, 1, *image_shape).to(device),
}

# Run SuperGlue
with torch.no_grad():
    pred = matching(data)

    matches = pred['matches0'][0].cpu().numpy()
    valid = matches > -1

    keypoints0 = data['keypoints0'][0].cpu().numpy()
    keypoints1 = data['keypoints1'][0].cpu().numpy()

    matched_kpts0 = keypoints0[valid]
    matched_kpts1 = keypoints1[matches[valid]]

# Draw matches with OpenCV
def draw_matches(img1, img2, kpts1, kpts2):
    img1 = np.array(img1.convert("RGB"))
    img2 = np.array(img2.convert("RGB"))

    kpts1_cv = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts1]
    kpts2_cv = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts2]
    matches_cv = [cv2.DMatch(i, i, 0) for i in range(len(kpts1))]
    
    matched_img = cv2.drawMatches(
        img1, kpts1_cv, img2, kpts2_cv, matches_cv, None,
        matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return matched_img


# Visualize matches
out_img = draw_matches(image1, image2, matched_kpts0, matched_kpts1)

plt.figure(figsize=(15, 8))
plt.imshow(out_img[..., ::-1])
plt.axis("off")
plt.title("SuperPoint + SuperGlue Matches")
plt.show()

# -----------------------------------------------------------
# Relative Pose Estimation using the matched keypoints
# -----------------------------------------------------------

# Convert matched keypoints to numpy arrays of type float32
pts1 = np.array(matched_kpts0, dtype=np.float32)
pts2 = np.array(matched_kpts1, dtype=np.float32)

# Define camera intrinsic parameters.
# Replace these values with your camera calibration data if available.
focal_length = 800  # Example focal length in pixels (adjust as needed)
principal_point = (image1.width / 2, image1.height / 2)

# Compute the Essential Matrix using RANSAC for robustness
E, mask_E = cv2.findEssentialMat(
    pts1, pts2,
    focal=focal_length,
    pp=principal_point,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)
print("Essential Matrix:\n", E)

# Recover the relative camera pose from the Essential Matrix
num_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, focal=focal_length, pp=principal_point)
print("Number of inliers:", num_inliers)
print("Rotation matrix R:\n", R)
print("Translation vector t:\n", t)

def select_keyframe(current_pose, last_keyframe_pose, translation_thresh=0.1, rotation_thresh=5):
    """
    Placeholder for keyframe selection logic.
    Compare current pose with last keyframe pose and decide if a new keyframe is needed.
    rotation_thresh: in degrees, translation_thresh: in appropriate scale units.
    """
    # For a simple example, select a new keyframe if translation difference is above a threshold.
    if last_keyframe_pose is None:
        return True
    # Compute translation difference (this is a simplistic check)
    translation_diff = np.linalg.norm(current_pose[1] - last_keyframe_pose[1])
    if translation_diff > translation_thresh:
        return True
    return False

