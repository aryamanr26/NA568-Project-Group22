import os
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import open3d as o3d
import gtsam
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from SuperGluePretrainedNetwork.models.matching import Matching  # from Magic Leap's SuperGlue repo
os.environ["QT_QPA_PLATFORM"] = "xcb"

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
    optimized_poses = [result.atPose3(i) for i in range(len(keyframes))]
    return optimized_poses

def match_features(feat1, feat2, image_shape):
    """
    Given features from two images, run SuperGlue to match keypoints.
    Returns matched keypoints arrays for image1 and image2.
    """
    data0 = format_features(feat1)
    data1 = format_features(feat2)
    
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
    
    with torch.no_grad():
        pred = matching_model(data)
    
    matches = pred['matches0'][0].cpu().numpy()
    valid = matches > -1
    keypoints0 = data['keypoints0'][0].cpu().numpy()
    keypoints1 = data['keypoints1'][0].cpu().numpy()
    matched_kpts0 = keypoints0[valid]
    matched_kpts1 = keypoints1[matches[valid]]
    return matched_kpts0, matched_kpts1

def estimate_pose(matched_kpts0, matched_kpts1, image_shape, focal_length=800):
    """
    Estimate the relative pose (R, t) between two frames using the Essential Matrix.
    """
    pts1 = np.array(matched_kpts0, dtype=np.float32)
    pts2 = np.array(matched_kpts1, dtype=np.float32)
    principal_point = (image_shape[1] / 2, image_shape[0] / 2)
    
    E, _ = cv2.findEssentialMat(pts1, pts2,
                                focal=focal_length,
                                pp=principal_point,
                                method=cv2.RANSAC,
                                prob=0.999,
                                threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=focal_length, pp=principal_point)
    return R, t, principal_point

def plot_poses_open3d(optimized_poses, frame_scale=0.5):
    """
    Visualize a list of optimized poses (gtsam.Pose3) using Open3D.
    
    Each pose is converted to a 4x4 transformation matrix. A coordinate frame is
    placed at each pose and the trajectory is drawn as a line connecting the translation points.
    
    Parameters:
        optimized_poses (list): List of gtsam.Pose3 objects.
        frame_scale (float): Scale of the coordinate frame axes.
    """
    points = []  # To store translation (position) points
    frames = []  # To store coordinate frames for each pose

    for pose in optimized_poses:
        # Convert gtsam.Pose3 to a 4x4 numpy array.
        # Depending on your gtsam python API, you might need to call .matrix() or .toMatrix()
        T = np.array(pose.matrix())
        points.append(T[:3, 3])
        
        # Create a coordinate frame and transform it with T.
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_scale)
        frame.transform(T)
        frames.append(frame)

    # Create a LineSet for the trajectory connecting the translation points.
    points = np.array(points)
    # Each line connects consecutive points
    lines = [[i, i+1] for i in range(len(points) - 1)]
    # Set line colors (red)
    colors = [[1, 0, 0] for _ in lines]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize the coordinate frames and the trajectory line.
    o3d.visualization.draw_geometries(frames + [line_set])

def triangulate_points(P1, P2, pts1, pts2):
    """Triangulate 3D points from two projection matrices and corresponding 2D points."""
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    pts4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d_hom[:3] / pts4d_hom[3]
    return pts3d.T

def plot_pose_trajectory(optimized_poses, plane="XZ"):
    """
    Plot the 2D trajectory from a list of gtsam.Pose3 objects.
    
    Parameters:
        optimized_poses (list): List of gtsam.Pose3 objects.
        plane (str): The plane to plot. Options are "XZ" (default), "XY", or "YZ".
    """
    import matplotlib.pyplot as plt
    
    x_coords = []
    y_coords = []
    
    for pose in optimized_poses:
        trans = pose.translation() if hasattr(pose, "translation") else pose[1]  # If pose is a tuple (R, t), use t

        if hasattr(trans, "x"):
            x_val = trans.x()
            y_val = trans.z() if plane.upper() == "XZ" else (trans.y() if plane.upper() == "XY" else trans.z())
        
        else:
         # Assume trans is a numpy array with shape (3,) or (3,1)
            trans = np.array(trans).flatten()
            x_val = trans[0]
            if plane.upper() == "XZ":
                y_val = trans[2]
            elif plane.upper() == "XY":
                y_val = trans[1]
            elif plane.upper() == "YZ":
                x_val = trans[1]
                y_val = trans[2]
            else:
                raise ValueError("Invalid plane. Choose from 'XZ', 'XY', or 'YZ'.")
        x_coords.append(x_val)
        y_coords.append(y_val)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-')
    plt.xlabel(plane[0])
    plt.ylabel(plane[1])
    plt.title(f'2D Pose Trajectory ({plane.upper()} Plane)')
    plt.grid(True)
    plt.show()

#############################################
# Main Execution
#############################################
if __name__ == '__main__':
    # Load images from local paths

    image_folder = "image_0"
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) == 0:
        print("No images found in the specified folder.")
        exit()

    focal_length = 800

    # Initial estimate
    prev_image = load_image(image_files[0])
    prev_feat = detect_and_extract(prev_image)
    image_shape = (prev_image.height, prev_image.width)
    
    # Initialize keyframe data with the first frame as keyframe (identity pose)
    keyframes = [(np.eye(3), np.zeros((3, 1)))]
    relative_poses = []
    last_keyframe_pose = keyframes[-1]
    
    # Process remaining images
    for path in image_files[1:100]:
        curr_image = load_image(path)
        curr_feat = detect_and_extract(curr_image)
        
        # Match features between previous keyframe and current frame
        matched_kpts0, matched_kpts1 = match_features(prev_feat, curr_feat, image_shape)
        if len(matched_kpts0) < 8:  # minimal matches required for pose estimation
            print(f"Not enough matches for image {path}, skipping.")
            continue
        
        # Estimate relative pose between previous keyframe and current frame
        R, t, principal_point = estimate_pose(matched_kpts0, matched_kpts1, image_shape, focal_length)
        current_pose = (R, t)
        
        # Check if current frame should be a keyframe
        if select_keyframe(current_pose, last_keyframe_pose):
            keyframes.append(current_pose)
            relative_poses.append(current_pose)
            last_keyframe_pose = current_pose
            print(f"Keyframe added: {path} (Total: {len(keyframes)})")
        
        # Update previous features with the current frame features
        # (Depending on your pipeline, you may choose to always match to the last keyframe,
        #  or use a sliding window, etc.)
        prev_feat = curr_feat

    # Pose Graph Optimization (if enough keyframes)
    if len(keyframes) > 1:
        optimized_poses = build_and_optimize_pose_graph(keyframes, relative_poses)
        print("Optimized Poses:")
        for i, pose in enumerate(optimized_poses):
            print(f"Keyframe {i}: {pose}")
    else:
        print("Not enough keyframes for pose graph optimization.")

    # Triangulation
    K = np.array([[focal_length, 0, principal_point[0]],
                  [0, focal_length, principal_point[1]],
                  [0, 0, 1]])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    pts3d = triangulate_points(P1, P2, matched_kpts0, matched_kpts1)
    print("Triangulated 3D points shape:", pts3d.shape)

    # plot_pose_trajectory(optimized_poses, plane="XZ")

    plot_poses_open3d(optimized_poses, frame_scale=0.5)
