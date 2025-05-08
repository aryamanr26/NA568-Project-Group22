import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import gtsam
from gtsam import Pose3, Rot3, Point3
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from SuperGluePretrainedNetwork.models.matching import Matching

# Ensure that Qt uses XCB on Linux; adjust as necessary for your environment.
os.environ["QT_QPA_PLATFORM"] = "xcb"


class SuperVisualOdometry:

    def __init__(self, image_folder, groundtruth_file, K, focal_length=707, translation_thresh=0.01, window_size = 10):
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
        self.window_size = window_size
        self.estimated_pose_text = []
        
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
                'sinkhorn_iterations': 100,
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
    
    def accumulate_relative_poses_gtsam(self, rel_poses, init_R, init_t):
        """
        Chain a list of relative poses into absolute poses using GTSAM Pose3.

        Args:
            rel_poses (list): List of tuples (R, t) for each relative movement.
            init_R (np.ndarray): Initial rotation (3×3).
            init_t (np.ndarray): Initial translation (3×1).
        Returns:
            abs_poses (list): List of accumulated (absolute) poses as (R, t).
        """
        # Initialize the running absolute Pose3
        pose_abs = Pose3(Rot3(init_R.copy()), Point3(*init_t.copy().flatten()))

        abs_poses = []
        for R_rel, t_rel in rel_poses:
            # Build a Pose3 from the relative (R, t)
            rel_pose = Pose3(Rot3(R_rel), Point3(*t_rel.flatten()))
            # Compose to get the new absolute pose
            pose_abs = pose_abs.compose(rel_pose)

            # Extract back to numpy
            R_abs = pose_abs.rotation().matrix()
            t_abs = pose_abs.translation().reshape(3, 1)

            abs_poses.append((R_abs, t_abs))

        return abs_poses
    
    def plot_pose_trajectory_single(self, optimized_poses, odometry_poses, groundtruth_poses, plane="XZ"):
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
        x_ot, y_ot = extract_coords(odometry_poses, plane)
        x_gt, y_gt = extract_coords(groundtruth_poses, plane)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        axis_labels = {"XZ": ("X", "Z"), "XY": ("X", "Y"), "YZ": ("Y", "Z")}
        xlabel, ylabel = axis_labels.get(plane.upper(), ("X", "Z"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x_est, y_est, marker='o', linestyle='-', label='Estimated')
        ax.plot(x_ot, y_ot, linestyle = '-', color = 'b', label = "Odometry")
        ax.plot(x_gt, y_gt, linestyle='--', color = 'tab:orange', label='Ground Truth')
        ax.set_title(f"2D Pose Trajectories ({plane.upper()} plane)")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def optimize_local_window(self):
        """
        Sliding-window BA on the last self.window_size keyframes.
        Updates self.keyframes in place.
        """
        n = len(self.keyframes)
        start = max(0, n - self.window_size)
        idxs = list(range(start, n))
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        sig_prior = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1e-6)
        sig_odom  = gtsam.noiseModel.Diagonal.Sigmas(np.full(6,0.05))

        # prior on first in window
        i0 = idxs[0]
        R0, t0 = self.keyframes[i0]
        p0 = Pose3(gtsam.Rot3(R0), t0.flatten())
        graph.add(gtsam.PriorFactorPose3(i0, p0, sig_prior))
        initial.insert(i0, p0)
        # insert rest
        for i in idxs[1:]:
            R,t = self.keyframes[i]
            initial.insert(i, Pose3(gtsam.Rot3(R), t.flatten()))

        # odom factors
        for k in range(start, n-1):
            Rr, tr = self.relative_poses[k]
            rel = Pose3(gtsam.Rot3(Rr), tr.flatten())
            graph.add(gtsam.BetweenFactorPose3(k, k+1, rel, sig_odom))

        opt = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
        for i in idxs:
            p = opt.atPose3(i)
            self.keyframes[i] = (p.rotation().matrix(), p.translation().reshape(3,1))
    
    def save_pose_kitti_format(self, R, t, pose_list):
       
        # Ensure t is a 1D array with 3 elements
        t = np.array(t).flatten()
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix R must have shape (3,3)")
        if t.shape != (3,):
            raise ValueError("Translation vector t must have 3 elements")
        
        # Create the 3x4 transformation matrix [R|t]
        T = np.hstack((R, t.reshape(3, 1)))
        
        # Flatten the matrix row-wise (resulting in 12 values)
        T_flat = T.flatten()
        
        # Format the values as scientific notation with 6 decimal places
        formatted_values = " ".join("{:.6e}".format(val) for val in T_flat)

        pose_list.append(formatted_values)

        return pose_list
    
    def write_strings_to_file(self, lines, filename):
        # Open the file in write mode. This will create the file if it doesn't exist.
        with open(filename, 'w') as file:
            for line in lines:
                # Write each string followed by a newline character.
                file.write(line + "\n")

    def perform_loop_closure(self, abs_poses):
        """
        Performs loop closure detection and graph optimization using GTSAM.
        
        This function constructs a factor graph from a list of absolute poses, adds odometry constraints 
        between consecutive poses, and includes loop closure factors when poses that are far apart in time 
        are close in space. After constructing the graph, it optimizes the poses using Levenberg-Marquardt.

        Args:
            abs_poses (list of tuple): Each element is a tuple (R, t) where:
                R (np.ndarray): Rotation matrix (3x3).
                t (np.ndarray): Translation vector (3, 1) or (3,).

        Returns:
            list of tuple: Optimized absolute poses as (R, t), where R is the rotation matrix 
                        and t is the translation vector in 1D numpy array form.
        """
        # Initialize factor graph and initial estimates container.
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()
        
        # Define noise models for different factors.
        noise_prior = gtsam.noiseModel.Diagonal.Variances(np.ones(6) * 1e-6)
        noise_odometry = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 0.05))
        noise_loop = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 0.03))
        
        # Process the first pose and add a prior factor.
        R0, t0 = abs_poses[0]
        # Flatten t0 if it is a column vector.
        t0 = t0.flatten() if t0.shape == (3, 1) else t0
        R0 = np.ascontiguousarray(R0, dtype=np.float64)
        t0 = np.ascontiguousarray(t0, dtype=np.float64)
        first_pose = gtsam.Pose3(gtsam.Rot3(R0), t0)
        graph.add(gtsam.PriorFactorPose3(0, first_pose, noise_prior))
        initial_estimates.insert(0, first_pose)
        
        # Add odometry factors between consecutive poses.
        for i in range(1, len(abs_poses)):
            R, t = abs_poses[i]
            t = t.flatten() if t.shape == (3, 1) else t
            current_pose = gtsam.Pose3(gtsam.Rot3(R), t)
            initial_estimates.insert(i, current_pose)
            
            # Compute relative transformation from the previous pose.
            R_prev, t_prev = abs_poses[i - 1]
            R_curr, t_curr = abs_poses[i]
            rel_R = R_prev.T @ R_curr
            rel_t = R_prev.T @ (t_curr - t_prev)
            rel_pose = gtsam.Pose3(gtsam.Rot3(rel_R), rel_t)
            graph.add(gtsam.BetweenFactorPose3(i - 1, i, rel_pose, noise_odometry))
        
        # Set hyper-parameter: minimum index difference to consider for loop closure.
        length_parameter = 500
        distance_variable = 200
        
        # Naive loop closure: connect poses that are spatially close yet far in sequence.
        for i in range(len(abs_poses)):
            for j in range(i + length_parameter, len(abs_poses), length_parameter):
                t_i = abs_poses[i][1]
                t_j = abs_poses[j][1]
                if np.linalg.norm(t_i - t_j) < distance_variable:  # Threshold set to 5 meters.
                    R_rel = abs_poses[i][0].T @ abs_poses[j][0]
                    t_rel = abs_poses[i][0].T @ (t_j - t_i)
                    loop_pose = gtsam.Pose3(gtsam.Rot3(R_rel), t_rel)
                    graph.add(gtsam.BetweenFactorPose3(i, j, loop_pose, noise_loop))
        
        # Optimize the factor graph.
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        result = optimizer.optimize()
        
        # Extract and return the optimized poses.
        optimized_poses = []
        for i in range(len(abs_poses)):
            pose_i = result.atPose3(i)
            R_i = pose_i.rotation().matrix()
            t_i = pose_i.translation()
            # Ensure translation is returned as a 1D numpy array.
            optimized_poses.append((R_i, np.array([t_i[0], t_i[1], t_i[2]])))
        
        return optimized_poses

    def run(self, length):
        """
        Execute the full pipeline:
        - Loads the first image and sets it as the initial keyframe.
        - Iterates over remaining images, matching features and estimating relative poses.
        - Selects keyframes based on the translation difference.
        - Accumulates relative poses into an overall trajectory.
        - Performs loop closure and graph optimization.
        - Plots the optimized trajectory against groundtruth.
        """
        first_image = self.load_image(self.image_files[0])
        first_feat = self.detect_and_extract(first_image)
        image_shape = (first_image.height, first_image.width)
        
        init_pose = (np.eye(3), np.zeros((3, 1)))
        R0, T0 = init_pose[0], init_pose[1]
        self.estimated_pose_text = self.save_pose_kitti_format(np.eye(3), np.zeros((3,1)), self.estimated_pose_text)
        self.absolute_pose = [init_pose]
        self.keyframes.append(init_pose)
        self.keyframe_gt.append(self.groundtruth_poses[0])
        last_keyframe_pose = init_pose
        prev_feat = first_feat
        
        for i, path in enumerate(self.image_files[1:length], start=1):
            curr_image = self.load_image(path)
            curr_feat = self.detect_and_extract(curr_image)
            
            # keep = curr_feat["scores"] > 0.5
            # curr_feat["keypoints"] = curr_feat["keypoints"][keep]
            # curr_feat["descriptors"] = curr_feat["descriptors"][keep]
            # curr_feat["scores"] = curr_feat["scores"][keep]

            # keep = prev_feat["scores"] > 0.5
            # prev_feat["keypoints"] = prev_feat["keypoints"][keep]
            # prev_feat["descriptors"] = prev_feat["descriptors"][keep]
            # prev_feat["scores"] = prev_feat["scores"][keep]

            matched_kpts0, matched_kpts1 = self.match_features(curr_feat, prev_feat, image_shape)
            if len(matched_kpts0) < 8:
                print(f"Not enough matches for image {path}, skipping.")
                # self.estimated_pose_text = self.save_pose_kitti_format(R, t, self.estimated_pose_text)
                continue
            
            R, t, _ = self.estimate_pose(matched_kpts0, matched_kpts1)
            current_pose = (R, t)
            # self.absolute_pose.append((self.absolute_pose[-1][0].dot(R), self.absolute_pose[-1][0].dot(t) + self.absolute_pose[-1][1]))
            # self.estimated_pose_text = self.save_pose_kitti_format(R, t, self.estimated_pose_text)
            # current_pose = self.absolute_pose[-1]
            

            if self.select_keyframe(current_pose, last_keyframe_pose):
                self.keyframes.append(current_pose)
                self.keyframe_gt.append(self.groundtruth_poses[i])
                self.relative_poses.append(current_pose)
                last_keyframe_pose = current_pose

                if len(self.keyframes) > 1:
                    self.optimize_local_window()

                print(f"Keyframe added: {path} (Total keyframes: {len(self.keyframes)})")
            
            prev_feat = curr_feat
        
        if len(self.keyframes) > 1:
            R_org = self.keyframes[0][0]
            t_org = self.keyframes[0][1]
            abs_poses = self.accumulate_relative_poses_gtsam(self.relative_poses, R_org, t_org)
            # abs_poses = self.absolute_pose
            abs_poses_optimized = self.perform_loop_closure(abs_poses)
            
            self.plot_pose_trajectory_single(abs_poses_optimized, abs_poses, self.keyframe_gt, plane="XZ")
            self.plot_pose_trajectory_single(abs_poses_optimized, abs_poses, self.keyframe_gt, plane="XY")
        else:
            print("Not enough keyframes for trajectory estimation.")

        for i, pose in enumerate(abs_poses):
            R, t = pose
            self.estimated_pose_text = self.save_pose_kitti_format(R, t, self.estimated_pose_text)
        # Saving estimated trajectory
        output_path = "/home/vtiaero/files/output/estimated_traj.txt"
        # self.write_strings_to_file(self.estimated_pose_text, output_path)
        # print(len(self.estimated_pose_text))



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
    
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) == 0:
        print("No images found in the specified folder.")
        exit()

    # Initialize and run the visual odometry system.
    vo_system = SuperVisualOdometry(image_folder, groundtruth_file, K,
                               focal_length=707, translation_thresh=0.01, window_size=20)
    iterations =  100 #len(image_files)
    vo_system.run(iterations)