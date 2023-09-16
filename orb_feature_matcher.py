import argparse
import time
import numpy as np
from lib.comms.orb_comms import ORBcomms
from eric_VO import VisualOdometry
import matplotlib.pyplot as plt


def plot_3d(pred_path):
    """
    Plots the 3D path

    Parameters
    ----------
    pred_path (ndarray): The predicted path
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pred_path[:, 0], pred_path[:, 2], pred_path[:,1])
    # make the scale of all the axis the same 
    max_range = np.array([pred_path[:, 0].max()-pred_path[:, 0].min(), pred_path[:, 2].max()-pred_path[:, 2].min(), pred_path[:, 1].max()-pred_path[:, 1].min()]).max() / 2.0
    mid_x = (pred_path[:, 0].max()+pred_path[:, 0].min()) * 0.5
    mid_y = (pred_path[:, 2].max()+pred_path[:, 2].min()) * 0.5
    mid_z = (pred_path[:, 1].max()+pred_path[:, 1].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def main():
    data_dir = 'KITTI_sequence_2'  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)
    oc = ORBcomms(mode="recv", ip="127.0.0.1", port=7000)


    start_time = time.time()
    estimated_path = []
    # loop until I press esc 
    cur_pose = None
    for i in range(0,50):
        print("i: ", i)
    
        keypoints1, descriptors1 = oc.receive_orb_data()
        if keypoints1 is None:
            continue
        q1, q2 = vo.get_mathces_new(descriptors1, keypoints1)
        if q1 is None:
            continue
        transf = vo.get_pose(q1, q2)
        if cur_pose is None:
            cur_pose = transf
        else:
            print("transf: ", transf)
            print("cur_pose: ", cur_pose)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        
        # print ("\nGround truth pose:\n" + str(gt_pose))
        # print ("\n Current pose:\n" + str(cur_pose))
        # print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        # gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]))
        
    print("Time taken: ", time.time() - start_time)
    #  calculate fps
    fps = len(vo.images) / (time.time() - start_time) 
    print("fps: ", fps)
  
    plot_3d(np.array(estimated_path))

if __name__ == "__main__":
    main()

