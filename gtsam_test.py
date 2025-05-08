import gtsam
import numpy as np

def test_gtsam():
    R = np.eye(3)
    t = np.zeros(3)
    try:
        rot = gtsam.Rot3(R)
        pose = gtsam.Pose3(rot, t)
        print("Pose3 created:", pose)
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    test_gtsam()
