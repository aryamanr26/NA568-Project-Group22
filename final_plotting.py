#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D pose trajectories from a .npz of pose arrays")
    parser.add_argument(
        "filename1", help="Path to .npz file (must contain ts_abs, ts_orb, ts_lp, ts_gt)")
    parser.add_argument(
        "filename2", help="Path to .npz file (must contain ts_abs, ts_orb, ts_lp, ts_gt)")
    parser.add_argument(
        "filename3", help="Path to .npz file (must contain ts_abs, ts_orb, ts_lp, ts_gt)")
    parser.add_argument(
        "filename4", help="Path to .npz file (must contain ts_abs, ts_orb, ts_lp, ts_gt)")
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
    data1 = np.load(args.filename1)
    data2 = np.load(args.filename2)
    data3 = np.load(args.filename3)
    data4 = np.load(args.filename4)

    # Rs_gt = data['Rs_gt']
    # Rs_abs = data['Rs_abs']
    # Rs_lp = data['Rs_lp']
    # Rs_orb = data['Rs_orb']
    
    ts_gt1 = data1['ts_gt']
    ts_abs1 = data1['ts_abs']  
    ts_orb1 = data1['ts_orb'] 
    ts_gt2 = data2['ts_gt']
    ts_abs2 = data2['ts_abs']  
    ts_orb2 = data2['ts_orb'] 
    ts_gt3 = data3['ts_gt']
    ts_abs3 = data3['ts_abs']  
    ts_orb3 = data3['ts_orb'] 
    ts_gt4 = data4['ts_gt']
    ts_abs4 = data4['ts_abs']  
    ts_orb4 = data4['ts_orb'] 
    # pick axes
    idx_map = {"XZ": (0, 2), "XY": (0, 1), "YZ": (1, 2)}
    xlabel, ylabel = {"XZ": ("X", "Z"), "XY": ("X", "Y"), "YZ": ("Y", "Z")}[args.plane]
    i, j = idx_map[args.plane]

    # extract 2D trajectories
    # x_est, y_est     = ts_lp[:, i],  ts_lp[:, j]
    # x_orb,  y_orb      = ts_orb1[:, i], ts_orb1[:, j]
    # x_odom, y_odom   = ts_abs[:, i], ts_abs[:, j]
    # x_gt,   y_gt     = ts_gt[:, i],   ts_gt[:, j]

    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # axes = axes.flatten()

    # for ax, (label, ts) in zip(axes, tsets.items()):
    #     x, y = ts[:, i], ts[:, j]
    #     ax.plot(x, y, marker='o', linestyle='-', label=label)
    #     ax.set_title(label)
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.grid(True)
    #     ax.axis('equal')
    #     ax.legend(loc='best')

    # plot
    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    ax0, ax1, ax2, ax3 = axes

        # --- Plot 1: Sequence 1 ---
    x_orb,  y_orb      = ts_orb1[:, i], ts_orb1[:, j]
    x_odom, y_odom   = ts_abs1[:, i], ts_abs1[:, j]
    x_gt,   y_gt     = ts_gt1[:, i],   ts_gt1[:, j]

    ax0.plot(x_orb, y_orb, color = "tab:green", linewidth = 2, linestyle='--', label='ORB')
    ax0.plot(x_odom, y_odom, color = "tab:red", linestyle='-', linewidth = 2, label='SuPer MVO')
    ax0.plot(x_gt, y_gt, color = "tab:blue", linewidth = 3, linestyle='-', label='Ground Truth')
    ax0.set_title(f"2D Pose Trajectories ({args.plane} plane): Sequence 00")
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.grid(True)
    ax0.axis('equal')
    ax0.legend(loc='best')

    # --- Plot 2: ORB Matches ---
    x_orb,  y_orb      = ts_orb2[:, i], ts_orb2[:, j]
    x_odom, y_odom   = ts_abs2[:, i], ts_abs2[:, j]
    x_gt,   y_gt     = ts_gt2[:, i],   ts_gt2[:, j]
    ax1.plot(x_orb, y_orb, color = "tab:green", linestyle='--', linewidth = 2, label='ORB')
    ax1.plot(x_odom, y_odom, color = "tab:red", linestyle='-', linewidth = 2, label='SuPerMVO')
    ax1.plot(x_gt, y_gt, color = "tab:blue", linewidth = 3, linestyle='-', label='Ground Truth')
    ax1.set_title(f"2D Pose Trajectories ({args.plane} plane): Sequence 06")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend(loc='best')

    # --- Plot 3: SuPerMVO ---
    x_orb,  y_orb      = ts_orb3[:, i], ts_orb3[:, j]
    x_odom, y_odom   = ts_abs3[:, i], ts_abs3[:, j]
    x_gt,   y_gt     = ts_gt3[:, i],   ts_gt3[:, j]
    ax2.plot(x_orb, y_orb, color = "tab:green", linestyle='--', linewidth = 2, label='ORB')
    ax2.plot(x_odom, y_odom, color = "tab:red", linestyle='-', linewidth = 2, label='SuPerMVO')
    ax2.plot(x_gt, y_gt, color = "tab:blue", linewidth = 3, linestyle='-', label='Ground Truth')
    ax2.set_title(f"2D Pose Trajectories ({args.plane} plane): Sequence 09")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend(loc='best')

    # --- Plot 4: Ground Truth ---
    x_orb,  y_orb      = ts_orb4[:, i], ts_orb4[:, j]
    x_odom, y_odom   = ts_abs4[:, i], ts_abs4[:, j]
    x_gt,   y_gt     = ts_gt4[:, i],   ts_gt4[:, j]
    ax3.plot(x_orb, y_orb, color = "tab:green", linestyle='--', linewidth = 2, label='ORB')
    ax3.plot(x_odom, y_odom, color = "tab:red", linewidth = 2, linestyle='-', label='SuPerMVO')
    ax3.plot(x_gt, y_gt, color = "tab:blue", linewidth = 3, linestyle='-', label='Ground Truth')
    ax3.set_title(f"2D Pose Trajectories ({args.plane} plane): Sequence 10")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    ax3.grid(True)
    ax3.lines[0].set_markersize(10)
    ax3.axis('equal')
    ax3.legend(loc='best')
    # e.g. highlight with a thicker marker:

    # finally, overall layout
    fig.suptitle(f"2D Pose Trajectories on the {args.plane} Plane", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #ax.plot(x_est, y_est, marker='o', color = "tab:blue", linestyle='-', label='Loop Closure')
    # ax.plot(x_lc, y_lc, marker='*', color = "tab:orange", linestyle='--', label='ORB')
    # ax.plot(x_odom, y_odom, marker='.', color = "tab:red", linestyle='-', label='SuperVO Odometry')
    # ax.plot(x_gt, y_gt, marker='_', color = "tab:green", linestyle='-', label='Ground Truth')

    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_title(f"2D Pose Trajectories ({args.plane} plane)")
    # ax.grid(True)
    # ax.axis('equal')
    # ax.legend(loc="best")
    # plt.tight_layout()
    # plt.show()
    fig.savefig("poster_graph_final", dpi = 1000, bbox_inches="tight")

if __name__ == "__main__":
    main()
