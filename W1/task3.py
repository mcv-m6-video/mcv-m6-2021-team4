import cv2




def task3_1(): 
    print("Task 3 - Quantitative evaluation of optical flow")

    ground_truth_path = "data/of_ground_truth"
    test_path = "data/of_estimations_LK"

    

def read_flow (file)

    im = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.double)

    # (u,v) flow vector
    im_u = (im[:, :, 2] - 2 ** 15) / 64
    im_v = (im[:, :, 1] - 2 ** 15) / 64

    # pixel exists or not
    im_valid = im[:, :, 0]
    im_valid[im_valid > 1] = 1

    im_u[im_valid == 0] = 0
    im_v[im_valid == 0] = 0

    flow_field = np.dstack((im_u, im_v, im_valid))

    return flow_field



if __name__ == "__main__":
    task3_1()