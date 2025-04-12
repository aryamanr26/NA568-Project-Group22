import os
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import open3d as o3d
import gtsam
from gtsam import Pose3
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

def select_keyframe(current_pose, last_keyframe_pose, translation_thresh=0.01):
    """Select a new keyframe based on translation difference."""
    if last_keyframe_pose is None:
        return True
    return np.linalg.norm(current_pose[1] - last_keyframe_pose[1]) > translation_thresh

def kitti_groundtruth(path):
    """
    Reads a KITTI groundtruth file where each line has 12 values (3x4 matrix).
    Returns:
      poses : list of (R, t) tuples
        R is 3x3 rotation, t is length-3 translation vector.
    """
    poses = []
    with open(path, "r") as f:
        # strip() to drop any trailing newline, splitlines() is cleaner
        for line in f.read().strip().splitlines():
            vals = np.fromstring(line, sep=' ')
            P = vals.reshape(3, 4)     # [ R | t ]
            R = P[:, :3]               # 3×3 rotation
            t = P[:, 3]                # 3×1 translation
            poses.append((R, t))
    return poses

def orb_vo_sequence_with_keyframes(img_paths, K,
                                   translation_thresh=0.1,
                                   nfeatures=2000):
    """
    ORB-VO with keyframe selection.
    - Matches each new frame to the last keyframe.
    - Updates keyframe when ||t_curr - t_last_kf|| > translation_thresh.

    Returns:
      poses : list of (R, t) global poses, one per input image
      keyframe_indices : list of ints, the frame indices chosen as keyframes
    """
    # your keyframe selector
    def select_keyframe(curr_pose, last_kf_pose):
        if last_kf_pose is None:
            return True
        return np.linalg.norm(curr_pose[1] - last_kf_pose[1]) > translation_thresh

    # init ORB & matcher
    orb = cv2.ORB_create(nfeatures)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # load first image, detect
    img0 = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    kp_kf, des_kf = orb.detectAndCompute(img0, None)

    # global pose starts at identity
    global_R = np.eye(3)
    global_t = np.zeros(3)

    poses = [(global_R.copy(), global_t.copy())]
    keyframe_indices = [0]
    last_kf_pose = (global_R.copy(), global_t.copy())

    # iterate through the rest
    for idx, p in enumerate(img_paths[1:], start=1):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)

        # match features last_kf → current
        matches = bf.knnMatch(des_kf, des, k=2)
        good = [m for m,n in matches if m.distance < 0.75 * n.distance]

        if len(good) >= 8:
            pts_kf = np.float32([kp_kf[m.queryIdx].pt for m in good])
            pts_cur = np.float32([kp[m.trainIdx].pt    for m in good])

            E, _        = cv2.findEssentialMat(
                pts_cur, pts_kf, K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            _, R_rel, t_rel, _ = cv2.recoverPose(E, pts_cur, pts_kf, K)
            t_rel = t_rel.flatten()
        else:
            # fallback: no motion
            R_rel = np.eye(3)
            t_rel = np.zeros(3)

        # ---- accumulate into global pose ----
        # new translation = old_t + old_R @ t_rel
        global_t = global_t + global_R.dot(t_rel)
        # new rotation = R_rel @ old_R
        global_R = R_rel.dot(global_R)

        poses.append((global_R.copy(), global_t.copy()))

        # ---- keyframe decision ----
        curr_pose = (global_R.copy(), global_t.copy())
        if select_keyframe(curr_pose, last_kf_pose):
            # update keyframe data
            kp_kf, des_kf = kp, des
            last_kf_pose  = curr_pose
            keyframe_indices.append(idx)

    return poses, keyframe_indices

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

def estimate_pose_K(pts1, pts2, K):
    """Estimate (R, t) from matched points and intrinsics K."""
    pts1, pts2 = [np.asarray(p, np.float32) for p in (pts1, pts2)]
    E, mask = cv2.findEssentialMat(
        pts1, pts2, cameraMatrix=K,
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K, mask=mask)
    return R, t, mask

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

def accumulate_relative_poses_rt(rel_poses, init_R=None, init_t=None):
    """
    Turn a list of relative poses into absolute poses by chaining them.
    Accepts rel_poses as either:
      - gtsam.Pose3 instances, or
      - tuples (R_rel, t_rel)

    Args:
        rel_poses: list of Pose3 or list of (R_rel, t_rel)
        init_R:    3x3 starting rotation (defaults to identity)
        init_t:    length-3 starting translation (defaults to zero)

    Returns:
        abs_poses: list of tuples (R_abs, t_abs) as NumPy arrays
    """
    # initialize
    last_R = init_R.copy() if init_R is not None else np.eye(3)
    last_t = init_t.copy() if init_t is not None else np.zeros(3)
    abs_poses = []

    for p in rel_poses:
        # unpack into R_rel, t_rel as numpy
        if isinstance(p, Pose3):
            R_rel = p.rotation().matrix()            # numpy (3×3)
            t3    = p.translation()                  # gtsam Point3
            t_rel = np.array([t3[0], t3[1], t3[2]])
        elif isinstance(p, tuple) and len(p) >= 2:
            R_rel = np.array(p[0])
            t_rel = np.array(p[1]).flatten()
        else:
            raise TypeError("Each element must be gtsam.Pose3 or (R, t) tuple")

        # chain
        R_abs = last_R.dot(R_rel)
        t_abs = last_R.dot(t_rel) + last_t

        abs_poses.append((R_abs, t_abs))

        last_R, last_t = R_abs, t_abs

    return abs_poses

def plot_pose_trajectory(optimized_poses, groundtruth_poses, plane="XZ"):
    """
    Plot 2D trajectories in two subplots:
      - left: optimized trajectory
      - right: ground-truth trajectory

    Parameters:
        optimized_poses (list): List of poses (gtsam.Pose3, (R,t) tuples, or raw 3‑vectors).
        groundtruth_poses (list): Same format as optimized_poses.
        plane (str): Which plane to plot: "XZ" (default), "XY", or "YZ".
    """

    def extract_coords(poses, plane):
        xs, ys = [], []
        for p in poses:
            # gtsam.Pose3?
            if hasattr(p, "translation"):
                t = p.translation()
                arr = np.array([t[0], t[1], t[2]])
            # (R, t) tuple?
            elif isinstance(p, tuple) and len(p) >= 2:
                arr = np.array(p[1]).flatten()
            # raw array-like
            else:
                arr = np.array(p).flatten()

            pl = plane.upper()
            if pl == "XZ":
                xs.append(arr[0]); ys.append(arr[2])
            elif pl == "XY":
                xs.append(arr[0]); ys.append(arr[1])
            elif pl == "YZ":
                xs.append(arr[1]); ys.append(arr[2])
            else:
                raise ValueError("Invalid plane: choose 'XZ', 'XY', or 'YZ'.")
        return xs, ys

    # extract coordinates
    x_est, y_est = extract_coords(optimized_poses, plane)
    x_gt,  y_gt  = extract_coords(groundtruth_poses, plane)

    # set up two subplots
    fig, (ax_est, ax_gt) = plt.subplots(1, 2, figsize=(12, 6))

    # axis labels
    axis_map = {"XZ": ("X", "Z"), "XY": ("X", "Y"), "YZ": ("Y", "Z")}
    xlabel, ylabel = axis_map.get(plane.upper(), ("X", "Z"))

    # plot optimized
    ax_est.plot(x_est, y_est, marker='o', linestyle='-')
    ax_est.set_title("Estimated Trajectory")
    ax_est.set_xlabel(xlabel)
    ax_est.set_ylabel(ylabel)
    ax_est.grid(True)
    ax_est.axis('equal')

    # plot ground truth
    ax_gt.plot(x_gt, y_gt, marker='x', linestyle='--')
    ax_gt.set_title("Ground Truth Trajectory")
    ax_gt.set_xlabel(xlabel)
    ax_gt.set_ylabel(ylabel)
    ax_gt.grid(True)
    ax_gt.axis('equal')

    # overall title
    fig.suptitle(f"2D Pose Trajectories ({plane.upper()} plane)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_pose_trajectory_single(optimized_poses,
                                groundtruth_poses,
                                plane="XZ"):
    """
    Plot 2D trajectories on a single plot:
      - optimized trajectory in solid circles
      - ground-truth trajectory in dashed crosses

    Parameters:
        optimized_poses (list): List of poses (gtsam.Pose3, (R,t) tuples, or raw 3‑vectors).
        groundtruth_poses (list): Same format as optimized_poses.
        plane (str): Which plane to plot: "XZ" (default), "XY", or "YZ".
    """

    def extract_coords(poses, plane):
        xs, ys = [], []
        for p in poses:
            # gtsam.Pose3?
            if hasattr(p, "translation"):
                t = p.translation()
                arr = np.array([t[0], t[1], t[2]])
            # (R, t) tuple?
            elif isinstance(p, tuple) and len(p) >= 2:
                arr = np.array(p[1]).flatten()
            # raw array-like
            else:
                arr = np.array(p).flatten()

            pl = plane.upper()
            if pl == "XZ":
                xs.append(arr[0]); ys.append(arr[2])
            elif pl == "XY":
                xs.append(arr[0]); ys.append(arr[1])
            elif pl == "YZ":
                xs.append(arr[1]); ys.append(arr[2])
            else:
                raise ValueError("Invalid plane: choose 'XZ', 'XY', or 'YZ'.")
        return xs, ys

    # extract coordinates
    x_est, y_est = extract_coords(optimized_poses, plane)
    x_gt,  y_gt  = extract_coords(groundtruth_poses, plane)

    # single plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # axis labels
    axis_map = {"XZ": ("X", "Z"), "XY": ("X", "Y"), "YZ": ("Y", "Z")}
    xlabel, ylabel = axis_map.get(plane.upper(), ("X", "Z"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # plot both trajectories
    ax.plot(x_est, y_est, marker='o', linestyle='-', label='Estimated')
    ax.plot(x_gt,  y_gt,  marker='x', linestyle='--', label='Ground Truth')

    ax.set_title(f"2D Pose Trajectories ({plane.upper()} plane)")
    ax.grid(True)
    ax.axis('equal')
    ax.legend()

    plt.tight_layout()
    plt.show()


#############################################
# Main Execution
#############################################
if __name__ == '__main__':
    # Load images from local paths
    fx, fy = 718.8560, 718.8560
    cx, cy = 607.1928, 185.2157
    K = np.array([[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]])
    kitti_groundtruth_path = "data_odometry_poses/dataset/poses/00.txt"
    image_folder = "image_0"

    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) == 0:
        print("No images found in the specified folder.")
        exit()

    focal_length = 707

    # Initial estimate
    prev_image = load_image(image_files[0])
    prev_feat = detect_and_extract(prev_image)
    image_shape = (prev_image.height, prev_image.width)
    
    # Initialize keyframe data with the first frame as keyframe (identity pose)
    keyframes = [(np.eye(3), np.zeros((3, 1)))]
    relative_poses = []
    last_keyframe_pose = keyframes[-1]
    
    ## Getting groundtruth poses from the KITTI Dataset
    groundtruth_poses = kitti_groundtruth(kitti_groundtruth_path)

    # Process remaining images
    keyframe_gt = [groundtruth_poses[0]]
    for i, path in enumerate(image_files[1:]):
        curr_image = load_image(path)
        curr_feat = detect_and_extract(curr_image)
        
        # Match features between previous keyframe and current frame
        matched_kpts0, matched_kpts1 = match_features(curr_feat, prev_feat, image_shape)
        # out_img = draw_matches(prev_image, curr_image, matched_kpts0, matched_kpts1)

        # plt.figure(figsize=(15, 8))
        # plt.imshow(out_img[..., ::-1])
        # plt.axis("off")
        # plt.title("SuperPoint + SuperGlue Matches")
        # plt.show()

        if len(matched_kpts0) < 8:  # minimal matches required for pose estimation
            print(f"Not enough matches for image {path}, skipping.")
            continue
        
        # Estimate relative pose between previous keyframe and current frame
        # R, t, principal_point = estimate_pose(matched_kpts0, matched_kpts1, image_shape, focal_length)
        R, t, principal_point = estimate_pose_K(matched_kpts0, matched_kpts1, K)
        current_pose = (R, t)
        
        # Check if current frame should be a keyframe
        if select_keyframe(current_pose, last_keyframe_pose):
            keyframes.append(current_pose)
            keyframe_gt.append(groundtruth_poses[i])
            relative_poses.append(current_pose)
            last_keyframe_pose = current_pose
            print(f"Keyframe added: {path} (Total: {len(keyframes)})")
            # print(f"Keyframe GroundTruth added: {path} (Total: {len(keyframe_gt)})")
        
        # Update previous features with the current frame features
        # (Depending on your pipeline, you may choose to always match to the last keyframe,
        #  or use a sliding window, etc.)
        prev_feat = curr_feat

    # Pose Graph Optimization (if enough keyframes)
    if len(keyframes) > 1:
        # optimized_poses = build_and_optimize_pose_graph(keyframes, relative_poses)
        optimized_poses = relative_poses
        print("Optimized Poses:")
        for i, pose in enumerate(optimized_poses):
            print(f"Keyframe {i}: {pose}")
    else:
        print("Not enough keyframes for pose graph optimization.")

    # print(optimized_poses[0])
    # print(optimized_poses[0].rotation().matrix())
    R_org = optimized_poses[0][0]
    t_org = np.array([optimized_poses[0][1][0], optimized_poses[0][1][1], optimized_poses[0][1][2]])
    rel_poses = accumulate_relative_poses_rt(optimized_poses, R_org, t_org)

    # R_org = optimized_poses[0].rotation().matrix()
    # t_org = np.array([optimized_poses[0].translation()[0], optimized_poses[0].translation()[1], optimized_poses[0].translation()[2]])
    # rel_poses = accumulate_relative_poses_rt(optimized_poses, R_org, t_org)

    # Triangulation
    # K = np.array([[focal_length, 0, principal_point[0]],
    #               [0, focal_length, principal_point[1]],
    #               [0, 0, 1]])
    # P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # P2 = K @ np.hstack((R, t))
    # pts3d = triangulate_points(P1, P2, matched_kpts0, matched_kpts1)
    # print("Triangulated 3D points shape:", pts3d.shape)

    # print(optimized_poses[0], type(optimized_poses[0]), optimized_poses[0].translation())
    # print(keyframe_gt[0], type(keyframe_gt[0]))
    
    plot_pose_trajectory_single(rel_poses, keyframe_gt, plane="XZ")
    plot_pose_trajectory_single(rel_poses, keyframe_gt, plane="XY")
    # plot_poses_open3d(optimized_poses, frame_scale=0.5)
