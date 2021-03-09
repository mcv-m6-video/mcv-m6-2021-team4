from os import path
import matplotlib.pyplot as plt
from flow_reader import read_flow
from flow_evaluation import compute_msen, compute_pepn


def task3_1_2(gt_path, estimated_path, frame):
    print("Task 3 - Quantitative evaluation of optical flow")

    gt = read_flow(path.join(gt_path, frame))
    estimated_flow = read_flow(path.join(estimated_path, "LKflow_" + frame))

    msen, sen = compute_msen(gt, estimated_flow)
    pepn = compute_pepn(gt, estimated_flow, sen)

    print(msen, pepn) # put the outputs nicer

    return msen, pepn, sen


def task3_3 (sen, frame):
    print("Task 3.3 - Visualization of Optical flow error")

    plt.hist(x=sen,bins=50)
    plt.savefig(frame)
    plt.clf()

    return


if __name__ == "__main__":

    gt_path = "../data/of_ground_truth"
    estimated_path = "../data/of_estimations_LK"
    frames = ["000045_10.png", "000157_10.png"]

    for frame in frames:
        msen, pepn, sen = task3_1_2(gt_path, estimated_path, frame)

        task3_3(sen, frame)
