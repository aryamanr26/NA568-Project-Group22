import os
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import gtsam
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from SuperGluePretrainedNetwork.models.matching import Matching  # from Magic Leap's SuperGlue repo

#############################################
# 1. Model Initialization
#############################################
print("Loading SuperPoint model...")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
sp_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Loading SuperGlue model (outdoor weights)...")
matching_model = Matching({
    'superpoint': {},
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}).eval().to(device)

#############################################
# Helper Functions
#############################################
def load_image(path):
    """Load an image from a file and convert it to RGB."""
    return Image.open(path).convert("RGB")

def detect_and_extract(image):
    """Run SuperPoint on a PIL image and return its features."""
    inputs = processor([image], return_tensors="pt")
    with torch.no_grad():
        outputs = sp_model(**inputs) ## unpacking dictionary into keyword arguments
    size = [(image.height, image.width)]
    features = processor.post_process_keypoint_detection(outputs, size)
    return features[0]

def format_features(feat):
    """Format SuperPoint features for SuperGlue input."""
    return {
        'keypoints': feat["keypoints"][None].to(device),
        'descriptors': feat["descriptors"][None].permute(0, 2, 1).to(device),
        'scores': feat["scores"][None].to(device)
    }

def draw_matches(img1, img2, kpts1, kpts2):
    """Draw keypoint matches between two images."""
    img1_np = np.array(img1.convert("RGB"))
    img2_np = np.array(img2.convert("RGB"))
    kpts1_cv = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts1]
    kpts2_cv = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts2]
    matches_cv = [cv2.DMatch(i, i, 0) for i in range(len(kpts1))]
    return cv2.drawMatches(img1_np, kpts1_cv, img2_np, kpts2_cv, matches_cv, None,
                           matchColor=(0, 255, 0),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def select_keyframe(current_pose, last_keyframe_pose, translation_thresh=0.1):
    """Select a new keyframe based on translation difference."""
    if last_keyframe_pose is None:
        return True
    return np.linalg.norm(current_pose[1] - last_keyframe_pose[1]) > translation_thresh

def build_and_optimize_pose_graph(keyframes, relative_poses):
    """Build and optimize a pose graph using GTSAM (pseudocode)."""
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1] * 6))
    first_pose = gtsam.Pose3()
    graph.add(gtsam.PriorFactorPose3(0, first_pose, pose_noise))
    initial_estimate.insert(0, first_pose)
    for i, (R, t) in enumerate(keyframes[1:], start=1):
        current_pose = gtsam.Pose3(gtsam.Rot3(R), t.flatten())
        initial_estimate.insert(i, current_pose)
        relative_pose = gtsam.Pose3(gtsam.Rot3(relative_poses[i - 1][0]),
                                    relative_poses[i - 1][1].flatten())
        graph.add(gtsam.BetweenFactorPose3(i - 1, i, relative_pose, pose_noise))
    parameters = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, parameters)
    result = optimizer.optimize()
    return [result.atPose3(i) for i in range(len(keyframes))]

def triangulate_points(P1, P2, pts1, pts2):
    """Triangulate 3D points from two projection matrices and corresponding 2D points."""
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    pts4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d_hom[:3] / pts4d_hom[3]
    return pts3d.T

#############################################
# Main Execution
#############################################
if __name__ == '__main__':
    # Load images from local paths
    img1_path = "image_0/000000.png"
    img2_path = "image_0/000005.png"

    image1 = load_image(img1_path)
    image2 = load_image(img2_path)
    images = [image1, image2]

    image_folder = "image_0"
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) == 0:
        print("No images found in the specified folder.")
        exit()


    # SuperPoint: Process images
    inputs = processor(images, return_tensors="pt")
    with torch.no_grad():
        outputs = sp_model(**inputs)
    sizes = [(img.height, img.width) for img in images]
    features = processor.post_process_keypoint_detection(outputs, sizes)

    data0 = format_features(features[0])
    data1 = format_features(features[1])
    image_shape = (image1.height, image1.width)
    data = {
        'keypoints0': data0['keypoints'],
        'keypoints1': data1['keypoints'],
        'descriptors0': data0['descriptors'],
        'descriptors1': data1['descriptors'],
        'scores0': data0['scores'],
        'scores1': data1['scores'],
        'image0': torch.empty(1, 1, *image_shape).to(device),
        'image1': torch.empty(1, 1, *image_shape).to(device),
    }

    # Run SuperGlue
    with torch.no_grad():
        pred = matching_model(data)
    matches = pred['matches0'][0].cpu().numpy()
    valid = matches > -1
    keypoints0 = data['keypoints0'][0].cpu().numpy()
    keypoints1 = data['keypoints1'][0].cpu().numpy()
    matched_kpts0 = keypoints0[valid]
    matched_kpts1 = keypoints1[matches[valid]]

    # Visualize matches
    out_img = draw_matches(image1, image2, matched_kpts0, matched_kpts1)
    plt.figure(figsize=(15, 8))
    plt.imshow(out_img[..., ::-1])
    plt.axis("off")
    plt.title("SuperPoint + SuperGlue Matches")
    plt.show()

    # Relative Pose Estimation
    pts1 = np.array(matched_kpts0, dtype=np.float32)
    pts2 = np.array(matched_kpts1, dtype=np.float32)
    focal_length = 800
    principal_point = (image1.width / 2, image1.height / 2)
    E, mask_E = cv2.findEssentialMat(pts1, pts2, focal=focal_length, pp=principal_point,
                                     method=cv2.RANSAC, prob=0.999, threshold=1.0)
    print("Essential Matrix:\n", E)
    num_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2,
                                                    focal=focal_length, pp=principal_point)
    print("Number of inliers:", num_inliers)
    print("Rotation matrix R:\n", R)
    print("Translation vector t:\n", t)

    # Keyframe Selection and Pose Graph Optimization
    keyframes = []
    relative_poses = []
    last_keyframe_pose = None
    current_pose = (R, t)
    if select_keyframe(current_pose, last_keyframe_pose):
        keyframes.append(current_pose)
        last_keyframe_pose = current_pose
    if len(keyframes) == 1:
        keyframes.append(current_pose)
        relative_poses.append(current_pose)
    optimized_poses = build_and_optimize_pose_graph(keyframes, relative_poses)
    print("Optimized Poses:", optimized_poses)

    # Triangulation
    K = np.array([[focal_length, 0, principal_point[0]],
                  [0, focal_length, principal_point[1]],
                  [0, 0, 1]])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    pts3d = triangulate_points(P1, P2, matched_kpts0, matched_kpts1)
    print("Triangulated 3D points shape:", pts3d.shape)
