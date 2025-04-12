import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import gtsam
from gtsam import Pose3
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from SuperGluePretrainedNetwork.models.matching import Matching

# Ensure that Qt uses XCB on Linux; adjust as necessary for your environment.
os.environ["QT_QPA_PLATFORM"] = "xcb"


class SuperVisualOdometry:
    def __init__(self, image_folder, groundtruth_file, K, focal_length=707, translation_thresh=0.01):
        """
        Initialize the visual odometry system.
        
        Args:
            image_folder (str): Path to the folder containing images.
            groundtruth_file (str): Path to KITTI-format groundtruth pose file.
            K (np.ndarray): Camera intrinsic matrix.
            focal_length (float): Focal length used in pose estimation.
            translation_thresh (float): Threshold for keyframe selection.
        """
        self.image_folder = image_folder
        self.groundtruth_file = groundtruth_file
        self.K = K
        self.focal_length = focal_length
        self.translation_thresh = translation_thresh
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load SuperPoint model & processor for keypoint extraction.
        print("Loading SuperPoint model...")
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.sp_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        
        # Load SuperGlue matching model with outdoor weights.
        print("Loading SuperGlue model (outdoor weights)...")
        matching_config = {
            'superpoint': {},
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matching_model = Matching(matching_config).eval().to(self.device)
        
        # Load groundtruth poses.
        self.groundtruth_poses = self.load_groundtruth(self.groundtruth_file)
        
        # Get a sorted list of image file paths.
        self.image_files = sorted([
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not self.image_files:
            raise ValueError("No images found in the specified folder.")
        
        # State initialization: keyframes, relative poses, and groundtruth keyframe poses.
        self.keyframes = []       # Each keyframe is a tuple (R, t)
        self.relative_poses = []  # Relative pose estimates between keyframes
        self.keyframe_gt = []     # Groundtruth poses associated with keyframes

    def load_groundtruth(self, path):
        """
        Load KITTI groundtruth poses from a file.
        
        Each line in the file should contain 12 values that represent the 3x4 pose matrix.
        Returns a list of tuples (R, t), where R is 3x3 and t is a 3x1 vector.
        """
        poses = []
        with open(path, "r") as f:
            for line in f.read().strip().splitlines():
                vals = np.fromstring(line, sep=' ')
                P = vals.reshape(3, 4)  # [R | t]
                R = P[:, :3]
                t = P[:, 3]
                poses.append((R, t))
        return poses

    def load_image(self, path):
        """Load an image from file and convert it to RGB."""
        return Image.open(path).convert("RGB")
    
    def detect_and_extract(self, image):
        """
        Use the SuperPoint model to detect keypoints and extract descriptors.
        
        Args:
            image (PIL.Image): The input image.
            
        Returns:
            A dictionary with keypoints, descriptors, and scores.
        """
        inputs = self.processor([image], return_tensors="pt")
        with torch.no_grad():
            outputs = self.sp_model(**inputs)
        size = [(image.height, image.width)]
        features = self.processor.post_process_keypoint_detection(outputs, size)
        return features[0]
    
    def format_features(self, feat):
        """
        Format the features into the structure required by SuperGlue.
        
        Args:
            feat (dict): Output of SuperPoint detection.
            
        Returns:
            A dictionary containing formatted keypoints, descriptors, and scores.
        """
        return {
            'keypoints': feat["keypoints"][None].to(self.device),
            'descriptors': feat["descriptors"][None].permute(0, 2, 1).to(self.device),
            'scores': feat["scores"][None].to(self.device)
        }
    
    def match_features(self, feat1, feat2, image_shape):
        """
        Use the SuperGlue model to match keypoints between two images.
        
        Args:
            feat1, feat2 (dict): Features from two images.
            image_shape (tuple): (height, width) of the image.
            
        Returns:
            matched_kpts0, matched_kpts1 (np.ndarray): Arrays of matched keypoints.
        """
        data0 = self.format_features(feat1)
        data1 = self.format_features(feat2)
        
        data = {
            'keypoints0': data0['keypoints'],
            'keypoints1': data1['keypoints'],
            'descriptors0': data0['descriptors'],
            'descriptors1': data1['descriptors'],
            'scores0': data0['scores'],
            'scores1': data1['scores'],
            'image0': torch.empty(1, 1, *image_shape).to(self.device),
            'image1': torch.empty(1, 1, *image_shape).to(self.device),
        }
        with torch.no_grad():
            pred = self.matching_model(data)
        matches = pred['matches0'][0].cpu().numpy()
        valid = matches > -1
        keypoints0 = data['keypoints0'][0].cpu().numpy()
        keypoints1 = data['keypoints1'][0].cpu().numpy()
        matched_kpts0 = keypoints0[valid]
        matched_kpts1 = keypoints1[matches[valid]]
        return matched_kpts0, matched_kpts1
    
    def estimate_pose(self, matched_kpts0, matched_kpts1):
        """
        Estimate the relative pose between two frames using the Essential matrix.
        
        Args:
            matched_kpts0, matched_kpts1 (np.ndarray): Matched keypoints from the two images.
            
        Returns:
            R (np.ndarray): Relative rotation.
            t (np.ndarray): Relative translation.
            principal_point (tuple): The principal point computed from the camera intrinsics.
        """
        pts1 = np.array(matched_kpts0, dtype=np.float32)
        pts2 = np.array(matched_kpts1, dtype=np.float32)
        principal_point = (self.K[0, 2], self.K[1, 2])
        
        E, _ = cv2.findEssentialMat(pts1, pts2, cameraMatrix=self.K,
                                    method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, cameraMatrix=self.K)
        return R, t, principal_point
    
    def select_keyframe(self, current_pose, last_keyframe_pose):
        """
        Decide whether the current frame qualifies as a keyframe.
        
        Args:
            current_pose (tuple): Pose (R, t) of the current frame.
            last_keyframe_pose (tuple): Pose (R, t) of the last keyframe.
        
        Returns:
            bool: True if the translation difference exceeds the threshold.
        """
        if last_keyframe_pose is None:
            return True
        return np.linalg.norm(current_pose[1] - last_keyframe_pose[1]) > self.translation_thresh
    
    def accumulate_relative_poses(self, rel_poses, init_R, init_t):
        """
        Chain a list of relative poses into absolute poses.
        
        Args:
            rel_poses (list): List of tuples (R, t) for each relative movement.
            init_R (np.ndarray): Initial rotation (usually identity).
            init_t (np.ndarray): Initial translation (usually zeros).
        
        Returns:
            abs_poses (list): List of accumulated (absolute) poses as (R, t).
        """
        last_R = init_R.copy()
        last_t = init_t.copy()
        abs_poses = []
        for R_rel, t_rel in rel_poses:
            R_abs = last_R.dot(R_rel)
            t_abs = last_R.dot(t_rel) + last_t
            abs_poses.append((R_abs, t_abs))
            last_R, last_t = R_abs, t_abs
        return abs_poses
    
    def plot_pose_trajectory_single(self, optimized_poses, groundtruth_poses, plane="XZ"):
        """
        Plot estimated and groundtruth 2D trajectories on a single plot.
        
        Args:
            optimized_poses (list): Estimated poses (each as (R, t)).
            groundtruth_poses (list): Groundtruth poses (each as (R, t)).
            plane (str): Which plane to plot ('XZ', 'XY', or 'YZ').
        """
        def extract_coords(poses, plane):
            xs, ys = [], []
            for pose in poses:
                # pose is a tuple (R, t)
                t = np.array(pose[1]).flatten()
                if plane.upper() == "XZ":
                    xs.append(t[0])
                    ys.append(t[2])
                elif plane.upper() == "XY":
                    xs.append(t[0])
                    ys.append(t[1])
                elif plane.upper() == "YZ":
                    xs.append(t[1])
                    ys.append(t[2])
                else:
                    raise ValueError("Invalid plane; choose 'XZ', 'XY', or 'YZ'.")
            return xs, ys

        x_est, y_est = extract_coords(optimized_poses, plane)
        x_gt, y_gt = extract_coords(groundtruth_poses, plane)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        axis_labels = {"XZ": ("X", "Z"), "XY": ("X", "Y"), "YZ": ("Y", "Z")}
        xlabel, ylabel = axis_labels.get(plane.upper(), ("X", "Z"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x_est, y_est, marker='o', linestyle='-', label='Estimated')
        ax.plot(x_gt, y_gt, marker='x', linestyle='--', label='Ground Truth')
        ax.set_title(f"2D Pose Trajectories ({plane.upper()} plane)")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def run(self, length):
        """
        Execute the full pipeline:
         - Loads the first image and sets it as the initial keyframe.
         - Iterates over remaining images, matching features and estimating relative poses.
         - Selects keyframes based on the translation difference.
         - Accumulates relative poses into an overall trajectory.
         - Plots the estimated trajectory against groundtruth.
        """
        # Initialize with the first frame.
        first_image = self.load_image(self.image_files[0])
        first_feat = self.detect_and_extract(first_image)
        image_shape = (first_image.height, first_image.width)
        
        # Set the first keyframe with an identity pose.
        init_pose = (np.eye(3), np.zeros((3, 1)))
        self.keyframes.append(init_pose)
        self.keyframe_gt.append(self.groundtruth_poses[0])
        last_keyframe_pose = init_pose
        prev_feat = first_feat
        
        # Process subsequent images.
        for i, path in enumerate(self.image_files[1: length], start=1):
            curr_image = self.load_image(path)
            curr_feat = self.detect_and_extract(curr_image)
            
            # Match features between the current frame and the last keyframe.
            matched_kpts0, matched_kpts1 = self.match_features(curr_feat, prev_feat, image_shape)
            if len(matched_kpts0) < 8:
                print(f"Not enough matches for image {path}, skipping.")
                continue
            
            # Estimate the relative pose.
            R, t, _ = self.estimate_pose(matched_kpts0, matched_kpts1)
            current_pose = (R, t)
            print(t)
            
            # Decide on keyframe selection based on the translation difference.
            if self.select_keyframe(current_pose, last_keyframe_pose):
                self.keyframes.append(current_pose)
                self.keyframe_gt.append(self.groundtruth_poses[i])
                self.relative_poses.append(current_pose)
                last_keyframe_pose = current_pose
                print(f"Keyframe added: {path} (Total keyframes: {len(self.keyframes)})")
            
            # Update the previous features (alternatively, you can always match to the last keyframe).
            prev_feat = curr_feat
        
        # If more than one keyframe was selected, accumulate the relative poses.
        if len(self.keyframes) > 1:
            R_org = self.keyframes[0][0]
            t_org = self.keyframes[0][1]
            abs_poses = self.accumulate_relative_poses(self.relative_poses, R_org, t_org)
            
            print("Accumulated Poses:")
            for idx, pose in enumerate(abs_poses):
                print(f"Pose {idx}:\nRotation:\n{pose[0]}\nTranslation: {pose[1]}\n")
            
            # Plot the estimated trajectory against the groundtruth.
            self.plot_pose_trajectory_single(abs_poses, self.keyframe_gt, plane="XZ")
            self.plot_pose_trajectory_single(abs_poses, self.keyframe_gt, plane="XY")
        else:
            print("Not enough keyframes for trajectory estimation.")


if __name__ == '__main__':
    # Define camera intrinsics (e.g., for the KITTI dataset).
    fx, fy = 718.8560, 718.8560
    cx, cy = 607.1928, 185.2157
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])
    
    # Paths to groundtruth file and image folder.
    groundtruth_file = "data_odometry_poses/dataset/poses/00.txt"
    image_folder = "image_0"
    
    # Initialize and run the visual odometry system.
    vo_system = SuperVisualOdometry(image_folder, groundtruth_file, K,
                               focal_length=707, translation_thresh=0.01)
    vo_system.run(10)
