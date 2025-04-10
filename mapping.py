import open3d as o3d
import numpy as np
import os
import cv2

class Visualizer3DMapping:
    def __init__(self, calib_file, pose_file, image_folder):
        self.calib_file = calib_file
        self.pose_file = pose_file
        self.image_folder = image_folder
        
        self.intrinsics = self.read_calib_file()
        self.poses = self.read_pose_file()

        # Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # List to store point clouds and trajectory
        self.point_clouds = []
        self.trajectory = []

    def read_calib_file(self):
        """Read camera intrinsics from calib file."""
        with open(self.calib_file, 'r') as f:
            for line in f:
                if line.startswith('P0'):
                    values = list(map(float, line.split()[1:]))
                    K = np.array(values).reshape(3, 4)[:3, :3]
                    return K
        raise ValueError("Calibration file does not contain 'P0'.")

    def read_pose_file(self):
        """Read the ground truth poses."""
        poses = []
        with open(self.pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.split()))
                pose = np.array(values).reshape(3, 4)
                poses.append(pose)
        return poses

    def read_single_image(self, index):
        """Read a single image."""
        image_files = sorted(os.listdir(self.image_folder))
        image_file = image_files[index]
        if image_file.endswith('.png'):
            img_path = os.path.join(self.image_folder, image_file)
            img = cv2.imread(img_path)
            return img
        return None

    def generate_sparse_point_cloud(self, image, pose, keypoints, descriptors):
        """Generate a sparse 3D point cloud from keypoints and pose."""
        pcd = o3d.geometry.PointCloud()

        if len(keypoints) == 0:
            print("No keypoints detected, skipping frame.")
            return pcd  # Empty point cloud if no keypoints detected

        # Create random 3D points (just for visualization purposes)
        points = np.random.rand(100, 3)  # Dummy 3D points
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def update_trajectory_and_pcd(self, pose, pcd):
        """Update the trajectory line and point cloud in the visualizer."""
        # Update trajectory
        self.trajectory.append(pose[:3, 3])

        # Add the point cloud to the visualizer
        self.vis.add_geometry(pcd)

        # Update the trajectory line (showing the path of poses)
        trajectory = np.array(self.trajectory)
        if len(trajectory) > 1:
            trajectory_line = o3d.geometry.LineSet()
            trajectory_line.points = o3d.utility.Vector3dVector(trajectory)
            trajectory_line.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(trajectory) - 1)])
            self.vis.add_geometry(trajectory_line)

        # Render updated geometry
        self.vis.update_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def visualize_mapping(self):
        """Visualize the 3D point cloud and the trajectory."""
        for i, pose in enumerate(self.poses):
            # Read a single image
            image = self.read_single_image(i)

            if image is None:
                print(f"Error: Image at index {i} could not be read.")
                continue

            # Detect keypoints and compute descriptors (using ORB for example)
            orb = cv2.ORB_create(nfeatures=1000)  # Increase the number of keypoints
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray_image, None)

            # Generate the sparse point cloud for this image
            pcd = self.generate_sparse_point_cloud(image, pose, keypoints, descriptors)

            # Update the visualizer with the point cloud and trajectory
            self.update_trajectory_and_pcd(pose, pcd)

            # Display the image in a separate window (for debugging purposes)
            # if i % 10 == 0:  # Display every 10th frame
            #     cv2.imshow(f"Image {i}", image)
            #     cv2.waitKey(1)  # 1 ms to allow window update

            # Update visualization after processing each frame
            self.vis.update_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

        # Final visualization
        self.vis.run()
        cv2.destroyAllWindows()

# Test code
if __name__ == "__main__":
    # Provide the paths to the files for sequence 0
    calib_file = './KITTI_dataset/dataset/sequences/00/calib.txt'
    pose_file = './poses/00.txt'
    image_folder = './KITTI_dataset/dataset/sequences/00/image_0'

    # Create the visualizer object
    visualizer = Visualizer3DMapping(calib_file, pose_file, image_folder)

    # Process and visualize the data
    visualizer.visualize_mapping()
