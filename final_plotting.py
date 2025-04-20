#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D pose trajectories from a .npz of pose arrays")
    parser.add_argument(
        "filename", help="Path to .npz file (must contain ts_abs, ts_orb, ts_lp, ts_gt)")
    parser.add_argument(
        "--plane",
        choices=("XZ", "XY", "YZ"),
        default="XZ",
        help="Which plane to project onto (default: XZ)")
    args = parser.parse_args()
    '''
    np.savez(
    filename,
    Rs_abs=Rs_abs,
    ts_abs=ts_abs,
    Rs_orb=Rs_orb,
    ts_orb=ts_orb,
    Rs_lp=Rs_lp,
    ts_lp=ts_lp,
    Rs_gt=Rs_gt,
    ts_gt=ts_gt
)
    '''
    # load translations
    data = np.load(args.filename)

    Rs_gt = data['Rs_gt']
    ts_gt = data['ts_gt']
    # Absolute Pose
    Rs_abs = data['Rs_abs']
    ts_abs = data['ts_abs']  

    Rs_lp = data['Rs_lp']
    ts_lp = data['ts_lp'] 

    Rs_orb = data['Rs_orb']
    ts_orb = data['ts_orb'] 

    # pick axes
    idx_map = {"XZ": (0, 2), "XY": (0, 1), "YZ": (1, 2)}
    xlabel, ylabel = {"XZ": ("X", "Z"), "XY": ("X", "Y"), "YZ": ("Y", "Z")}[args.plane]
    i, j = idx_map[args.plane]

    # extract 2D trajectories
    x_est, y_est     = ts_lp[:, i],  ts_lp[:, j]
    x_lc,  y_lc      = ts_orb[:, i], ts_orb[:, j]
    x_odom, y_odom   = ts_abs[:, i], ts_abs[:, j]
    x_gt,   y_gt     = ts_gt[:, i],   ts_gt[:, j]

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_est, y_est, marker='o', color = "tab:blue", linestyle='-', label='Loop Closure')
    ax.plot(x_lc, y_lc, marker='*', color = "tab:orange", linestyle='--', label='ORB')
    ax.plot(x_odom, y_odom, marker='.', color = "tab:red", linestyle='-', label='SuperVO Odometry')
    ax.plot(x_gt, y_gt, marker='_', color = "tab:green", linestyle='-', label='Ground Truth')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"2D Pose Trajectories ({args.plane} plane)")
    ax.grid(True)
    ax.axis('equal')
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
