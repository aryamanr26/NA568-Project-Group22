import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_12_to_tum(input_file, output_file, dt=0.1):
    """
    Convert a file of 12-value rows (3x4 pose matrices) into TUM format.

    Args:
        input_file (str): Path to the input file. Each line must have 12 floats.
        output_file (str): Path for the TUM-format output file.
        dt (float): Time difference between consecutive frames (in seconds).
                    Used to generate synthetic timestamps as idx * dt.
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for idx, line in enumerate(f_in):
            # parse 12 floats
            vals = np.fromstring(line, sep=' ')
            if vals.size != 12:
                raise ValueError(f"Line {idx+1} in {input_file} does not have 12 values")
            # reshape into 3×4
            T = vals.reshape(3, 4)
            rot_mat = T[:, :3]
            trans = T[:, 3]

            # convert rotation matrix to quaternion (x, y, z, w)
            quat = R.from_matrix(rot_mat).as_quat()

            timestamp = idx * dt
            # write: timestamp tx ty tz qx qy qz qw
            f_out.write(f"{timestamp:.6f} "
                        f"{trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} "
                        f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

# --- example usage ---
# convert_12_to_tum("kitti_poses.txt", "kitti_tum.txt", dt=0.1)
# convert_12_to_tum("data_odometry_poses/dataset/poses/00.txt", "kitti_gt_tum.txt")

def save_pose_kitti_format(R, t, pose_list):
    
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

def write_strings_to_file(lines, filename):
        # Open the file in write mode. This will create the file if it doesn't exist.
        with open(filename, 'w') as file:
            for line in lines:
                # Write each string followed by a newline character.
                file.write(line + "\n")
if __name__ == "__main__":
    psuedo = 0
    if psuedo:
        ## SPECIFICALLY FOR CONVERTING ORB DATA INTO TUM FILE
        filename = "final_numpy/seq_10_plot.npz"
        seq = 10
        evo_est_filename = "evo/est{}.txt".format(seq)
        evo_orb_filename = "evo/orb{}.txt".format(seq)

        data = np.load(filename)
        Ro = data["Rs_orb"]
        to = data["ts_orb"]
        print(Ro[0])
        print(to[0])
        pose_list = []
        for i, (r, t) in enumerate(zip(Ro,to)):
            pose_list = save_pose_kitti_format(r, t, pose_list)
        write_strings_to_file(pose_list, evo_orb_filename)

        Ro = data["Rs_abs"]
        to = data["ts_abs"]
        print(Ro[0])
        print(to[0])
        pose_list = []
        for i, (r, t) in enumerate(zip(Ro,to)):
            pose_list = save_pose_kitti_format(r, t, pose_list)
        write_strings_to_file(pose_list, evo_est_filename)
        # orb_file = "orb_data.txt"
        # write_strings_to_file(pose_list, orb_file)
        evo_est_tum_filename = "evo/tum_est{}.txt".format(seq)
        evo_orb_tum_filename = "evo/tum_orb{}.txt".format(seq)
        # convert_12_to_tum("output/estimated_traj.txt", "kitti_est_tum_2.txt")
        convert_12_to_tum(evo_orb_filename, evo_orb_tum_filename)
        convert_12_to_tum(evo_est_filename, evo_est_tum_filename)

    else:
        groundtruth = "data_odometry_poses/dataset/poses/10.txt"
        groundtruth_tum = "evo/tum_gt_10.txt"
        convert_12_to_tum(groundtruth, groundtruth_tum)