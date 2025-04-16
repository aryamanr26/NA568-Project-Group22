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

        # ========== New: Initialize BoW vocabulary for loop closure ==========
        self.detector = cv2.SIFT_create()   # Keypoint detector
        self.extractor = cv2.SIFT_create()  # Descriptor extractor
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        # Separate BoW trainer and extractor
        self.bow_trainer = cv2.BOWKMeansTrainer(200)  # Use this to collect descriptors
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.extractor, self.matcher)  # Use this after vocab is built

        self.vocab_built = False
        self.bow_descriptors = []  # Store BoW histograms for loop closure

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
        ax.plot(x_ot, y_ot, marker='*', linestyle='-', label='Odometry')
        ax.plot(x_gt, y_gt, marker='x', linestyle='--', label='Ground Truth')
        ax.set_title(f"2D Pose Trajectories ({plane.upper()} plane)")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    # def perform_loop_closure(self, abs_poses):
    #     """
    #     Performs simple loop closure detection and graph optimization using GTSAM.
        
    #     Args:
    #         abs_poses (list): List of absolute poses [(R, t), ...]
        
    #     Returns:
    #         optimized_poses: List of optimized absolute poses after loop closure correction.
    #     """
    #     # print(1)
    #     graph = gtsam.NonlinearFactorGraph()
    #     # print(2)
    #     initial_estimates = gtsam.Values()
    #     # print(3)
    #     noise_prior = gtsam.noiseModel.Diagonal.Variances(np.ones(6) * 1e-6)
    #     noise_odometry = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))
    #     noise_loop = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.03]*6))
    #     # print(4)

    #     # Add prior on the first pose
    #     # print(abs_poses[0])
    #     R0, t0 = abs_poses[0]
    #     # print(R0.shape)
    #     # print(t0.shape)

    #     # Sanity check on R0
    #     should_be_identity = R0.T @ R0
    #     det = np.linalg.det(R0)
    #     # print("R0ᵀ * R0 =\n", should_be_identity)
    #     # print("det(R0) =", det)

    #     # if not np.allclose(should_be_identity, np.eye(3), atol=1e-3):
    #     #     raise ValueError("❌ R0 is not orthogonal!")

    #     # if not np.isclose(det, 1.0, atol=1e-3):
    #     #     raise ValueError("❌ det(R0) ≠ 1. Invalid rotation matrix!")

    #     if t0.shape == (3, 1): t0 = t0.flatten()
    #     R0 = np.ascontiguousarray(R0, dtype=np.float64)
    #     t0 = np.ascontiguousarray(t0, dtype=np.float64)
    #     first_pose = gtsam.Pose3(gtsam.Rot3(R0), t0)
    #     # print(5)
    #     # print(first_pose)
    #     graph.add(gtsam.PriorFactorPose3(0, first_pose, noise_prior))
    #     # print(6)
    #     initial_estimates.insert(0, first_pose)
    #     # print(7)

    #     # Add odometry constraints
    #     for i in range(1, len(abs_poses)):
    #         R, t = abs_poses[i]
    #         if t.shape == (3, 1): t = t.flatten()
    #         pose_i = gtsam.Pose3(gtsam.Rot3(R), t)
    #         initial_estimates.insert(i, pose_i)

    #         R_prev, t_prev = abs_poses[i-1]
    #         R_curr, t_curr = abs_poses[i]
    #         rel_R = R_prev.T @ R_curr
    #         rel_t = R_prev.T @ (t_curr - t_prev)
    #         rel_pose = gtsam.Pose3(gtsam.Rot3(rel_R), rel_t)
    #         graph.add(gtsam.BetweenFactorPose3(i-1, i, rel_pose, noise_odometry))

    #     # Hyper parameter for loop closure distance
    #     length_parameter = 1000 
        
    #     # Naive loop closure: connect current frame to earlier if it's close in space
    #     for i in range(len(abs_poses)):
    #         for j in range(i+length_parameter, len(abs_poses), length_parameter):  # Skip nearby frames
    #             t_i = abs_poses[i][1]
    #             t_j = abs_poses[j][1]
    #             dist = np.linalg.norm(t_i - t_j)
    #             if dist < 5:  # Arbitrary threshold for loop detection
    #                 R_rel = abs_poses[i][0].T @ abs_poses[j][0]
    #                 t_rel = abs_poses[i][0].T @ (t_j - t_i)
    #                 loop_pose = gtsam.Pose3(gtsam.Rot3(R_rel), t_rel)
    #                 graph.add(gtsam.BetweenFactorPose3(i, j, loop_pose, noise_loop))

    #     # Optimize
    #     optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
    #     result = optimizer.optimize()

    #     # Extract optimized poses
    #     optimized_poses = []
    #     for i in range(len(abs_poses)):
    #         pose_i = result.atPose3(i)
    #         R_i = pose_i.rotation().matrix()
    #         t_i = pose_i.translation()
    #         optimized_poses.append((R_i, np.array([t_i[0], t_i[1], t_i[2]])))
        
    #     return optimized_poses

    def perform_loop_closure(self, abs_poses):
        """
        Performs loop closure detection using BoW similarity and graph optimization using GTSAM.
        
        Args:
            abs_poses (list): List of absolute poses [(R, t), ...]
        
        Returns:
            optimized_poses: List of optimized absolute poses after loop closure correction.
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()

        noise_prior = gtsam.noiseModel.Diagonal.Variances(np.ones(6) * 1e-6)
        noise_odometry = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))
        noise_loop = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.03]*6))

        # Add prior on the first pose
        R0, t0 = abs_poses[0]
        if t0.shape == (3, 1): t0 = t0.flatten()
        R0 = np.ascontiguousarray(R0, dtype=np.float64)
        t0 = np.ascontiguousarray(t0, dtype=np.float64)
        first_pose = gtsam.Pose3(gtsam.Rot3(R0), t0)
        graph.add(gtsam.PriorFactorPose3(0, first_pose, noise_prior))
        initial_estimates.insert(0, first_pose)

        # Add odometry constraints
        for i in range(1, len(abs_poses)):
            R, t = abs_poses[i]
            if t.shape == (3, 1): t = t.flatten()
            pose_i = gtsam.Pose3(gtsam.Rot3(R), t)
            initial_estimates.insert(i, pose_i)

            R_prev, t_prev = abs_poses[i-1]
            R_curr, t_curr = abs_poses[i]
            rel_R = R_prev.T @ R_curr
            rel_t = R_prev.T @ (t_curr - t_prev)
            rel_pose = gtsam.Pose3(gtsam.Rot3(rel_R), rel_t)
            graph.add(gtsam.BetweenFactorPose3(i-1, i, rel_pose, noise_odometry))

        # === BoW-based loop closure detection ===
        print("🔁 Performing BoW-based loop closure detection...")
        for i in range(len(abs_poses)):
            desc_i = self.bow_descriptors[i]
            if desc_i is None:
                continue

            best_j = -1
            best_score = float('inf')

            # Search backward for most visually similar frame
            for j in range(i - 1):
                desc_j = self.bow_descriptors[j]
                if desc_j is None:
                    continue
                score = np.linalg.norm(desc_i - desc_j)
                if score < best_score:
                    best_score = score
                    best_j = j

            # Add a loop closure if similarity is high
            if best_j != -1 and best_score < 0.5:  # Threshold tuned experimentally
                print(f"📎 Loop closure added between {i} and {best_j}, score={best_score:.3f}")
                R_i, t_i = abs_poses[i]
                R_j, t_j = abs_poses[best_j]
                R_rel = R_j.T @ R_i
                t_rel = R_j.T @ (t_i - t_j)
                loop_pose = gtsam.Pose3(gtsam.Rot3(R_rel), t_rel)
                graph.add(gtsam.BetweenFactorPose3(best_j, i, loop_pose, noise_loop))

        # === Graph optimization ===
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        result = optimizer.optimize()

        # === Extract optimized poses ===
        optimized_poses = []
        for i in range(len(abs_poses)):
            pose_i = result.atPose3(i)
            R_i = pose_i.rotation().matrix()
            t_i = pose_i.translation()
            optimized_poses.append((R_i, np.array([t_i[0], t_i[1], t_i[2]])))

        return optimized_poses

    def compute_bow_descriptor(self, image):
        """
        Compute BoW descriptor for a given image using SIFT features.
        Optimized for speed and memory during large-scale clustering.
        """
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        keypoints = self.detector.detect(gray, None)

        # Initialize bookkeeping attributes
        if not hasattr(self, "keyframe_index"):
            self.keyframe_index = 0
        if not hasattr(self, "total_desc_count"):
            self.total_desc_count = 0

        # Collect descriptors only before vocabulary is built
        if not self.vocab_built:
            _, descriptors = self.extractor.compute(gray, keypoints)

            if (
                descriptors is not None and
                self.keyframe_index % 10 == 0 and
                self.total_desc_count < 30000
            ):
                # Limit descriptors per frame
                max_desc = 300
                if descriptors.shape[0] > max_desc:
                    descriptors = descriptors[:max_desc]

                self.bow_trainer.add(descriptors)
                self.total_desc_count += descriptors.shape[0]

            self.keyframe_index += 1
            return None  # Don’t compute histogram yet

        # Vocabulary is ready — compute BoW descriptor
        return self.bow_extractor.compute(gray, keypoints)

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
            # print(t)
            
            # Decide on keyframe selection based on the translation difference.
            if self.select_keyframe(current_pose, last_keyframe_pose):
                self.keyframes.append(current_pose)
                self.keyframe_gt.append(self.groundtruth_poses[i])
                self.relative_poses.append(current_pose)
                last_keyframe_pose = current_pose
                print(f"Keyframe added: {path} (Total keyframes: {len(self.keyframes)})")
            
                # === New: Compute and store BoW descriptor ===
                bow_desc = self.compute_bow_descriptor(curr_image)
                self.bow_descriptors.append(bow_desc)
            
            # Update the previous features (alternatively, you can always match to the last keyframe).
            prev_feat = curr_feat
        
        # === Step 3: Build Vocabulary (After all keyframes collected) ===
        print("Building BoW Vocabulary...")
        vocab = self.bow_trainer.cluster()  # ✅ Now correctly uses the trainer
        self.bow_extractor.setVocabulary(vocab)
        self.vocab_built = True

        # If more than one keyframe was selected, accumulate and optimize the trajectory.
        if len(self.keyframes) > 1:
            R_org = self.keyframes[0][0]
            t_org = self.keyframes[0][1]
            abs_poses = self.accumulate_relative_poses(self.relative_poses, R_org, t_org)
            
            print("Accumulated Poses:")
            # for idx, pose in enumerate(abs_poses):
            #     print(f"Pose {idx}:\nRotation:\n{pose[0]}\nTranslation: {pose[1]}\n")
            
            # LOOP CLOSURE: refine poses with GTSAM optimization

            abs_poses_optimized = self.perform_loop_closure(abs_poses)
            # print("I'm here")
          
            est_xyz = np.array([pose[1].flatten() for pose in abs_poses_optimized])
            gt_xyz = np.array([pose[1].flatten() for pose in self.keyframe_gt])
            # Match lengths before alignment
            min_len = min(len(est_xyz), len(gt_xyz))
            est_xyz = est_xyz[:min_len]
            gt_xyz = gt_xyz[:min_len]
            aligned_xyz = align_trajectories_umeyama(est_xyz, gt_xyz)

            # Reconstruct aligned pose list (reuse original rotation)
            aligned_poses = [(abs_poses_optimized[i][0], aligned_xyz[i]) for i in range(len(est_xyz))]


            # Plot the optimized trajectory against the groundtruth.
            self.plot_pose_trajectory_single(aligned_poses, abs_poses, self.keyframe_gt, plane="XZ")
            self.plot_pose_trajectory_single(aligned_poses, abs_poses, self.keyframe_gt, plane="XY")
        else:
            print("Not enough keyframes for trajectory estimation.")


def align_trajectories_umeyama(estimated_xyz, gt_xyz):
    """
    Aligns estimated trajectory to ground truth using Umeyama similarity transform.
    
    Args:
        estimated_xyz (np.ndarray): Nx3 estimated positions
        gt_xyz (np.ndarray): Nx3 ground truth positions
    
    Returns:
        aligned_xyz (np.ndarray): Nx3 aligned estimated positions
    """
    assert estimated_xyz.shape == gt_xyz.shape, "Shape mismatch for alignment"

    N = estimated_xyz.shape[0]

    # Convert to Open3D point clouds
    source_pc = o3d.geometry.PointCloud()
    target_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(estimated_xyz)
    target_pc.points = o3d.utility.Vector3dVector(gt_xyz)

    # Construct 1-to-1 correspondences
    correspondences = o3d.utility.Vector2iVector(np.array([[i, i] for i in range(N)]))

    # Estimate transformation with scaling
    trans = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    transformation = trans.compute_transformation(source_pc, target_pc, correspondences)

    # Apply transformation
    aligned_xyz = np.asarray(source_pc.points) @ transformation[:3, :3].T + transformation[:3, 3]

    return aligned_xyz

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
                               focal_length=707, translation_thresh=0.01)
    n = 1000 # len(image_files)
    vo_system.run(n)