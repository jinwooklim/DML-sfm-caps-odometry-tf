#!/usr/bin/env python
__author__ = "JinWook Lim"

import numpy as np
import math
import pandas as pd
import os
import glob

def make_class_and_error(degree):
    '''
    mean = (-55.0 + -45.0)/2.0
    error_degree = (radian - mean) / 10.0
    :param degree:
    :return: data_class, error_degree:
    '''

    if(-55.0 <= degree and degree < -45.0):
        error_degree = (degree - ((-55.0 + -45.0) / 2.0)) / 10.0
        data_class = 0
        return data_class, error_degree
    elif (-45.0 <= degree and degree < -35.0):
        error_degree = (degree - ((-45.0 + -35.0) / 2.0)) / 10.0
        data_class = 1
        return data_class, error_degree
    elif (-35.0 <= degree and degree < -25.0):
        error_degree = (degree - ((-35.0 + -25.0) / 2.0)) / 10.0
        data_class = 2
        return data_class, error_degree
    elif (-25.0 <= degree and degree < -15.0):
        error_degree = (degree - ((-25.0 + -15.0) / 2.0)) / 10.0
        data_class = 3
        return data_class, error_degree
    elif (-15.0 <= degree and degree < -5.0):
        error_degree = (degree - ((-15.0 + -5.0) / 2.0)) / 10.0
        data_class = 4
        return data_class, error_degree
    elif (-5.0 <= degree and degree <= 5.0):
        error_degree = (degree - ((-5.0 + 5.0) / 2.0)) / 10.0
        data_class = 5
        return data_class, error_degree
    elif (5.0 < degree and degree <= 15.0):
        error_degree = (degree - ((5.0 + 15.0) / 2.0)) / 10.0
        data_class = 6
        return data_class, error_degree
    elif (15.0 < degree and degree <= 25.0):
        error_degree = (degree - ((15.0 + 25.0) / 2.0)) / 10.0
        data_class = 7
        return data_class, error_degree
    elif (25.0 < degree and degree <= 35.0):
        error_degree = (degree - ((25.0 + 35.0) / 2.0)) / 10.0
        data_class = 8
        return data_class, error_degree
    elif (35.0 < degree and degree <= 45.0):
        error_degree = (degree - ((35.0 + 45.0) / 2.0)) / 10.0
        data_class = 9
        return data_class, error_degree
    elif (45.0 < degree and degree <= 55.0):
        error_degree = (degree - ((45.0 + 55.0) / 2.0)) / 10.0
        data_class = 10
        return data_class, error_degree

def generate_data(pose_path):
    pose_file = np.loadtxt(pose_path, delimiter=' ') # 4541x12
    row_size = np.size(pose_file, 0)

    # Make 3x3x4 matrix list
    index_list = []
    R_list = []
    t_list = []
    tx_list = []
    ty_list = []
    tz_list = []
    for i in range(row_size):
        index_list.append("%06d"%(i+1))
        R = np.array([[pose_file[i][0]],
                      [pose_file[i][1]],
                      [pose_file[i][2]],
                      [pose_file[i][4]],
                      [pose_file[i][5]],
                      [pose_file[i][6]],
                      [pose_file[i][8]],
                      [pose_file[i][9]],
                      [pose_file[i][10]]])
        t = np.array([[pose_file[i][3]],
                      [pose_file[i][7]],
                      [pose_file[i][11]]])
        R = np.reshape(R, (3, 3))
        t = np.reshape(t, (3, 1))
        R_list.append(R)
        t_list.append(t)
        tx_list.append(t[0][0])
        ty_list.append(t[1][0])
        tz_list.append(t[2][0])

    # Let tx, ty, tz have relative values from the previous frame
    # If delete this part, they have relative values from the init frame
    # We don't need modify ty
    '''
    for i in range(len(tx_list)):
        if(i == 0):
            tx_list[i] = tx_list[i]
            ty_list[i] = ty_list[i]
            tz_list[i] = tz_list[i]
        else:
            tx_list[i] = tx_list[i] - tx_list[i-1]
            ty_list[i] = ty_list[i]
            tz_list[i] = tz_list[i]
    '''


    # Find Degree and then, get data_class and error_degree
    yaw_list = []
    data_class_list = []
    error_degree_list = []
    for i in range(row_size):
        R = R_list[i]
        # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
        # https://stackoverflow.com/questions/11514063/extract-yaw-pitch-and-roll-from-a-rotationmatrix
        yaw_rate = -math.atan2(R[0, 2], R[1, 2]) # -arctan2(A13, A23)
        yaw_list.append(yaw_rate)
        data_class, error_degree = make_class_and_error(yaw_rate)
        data_class_list.append(data_class)
        error_degree_list.append(error_degree)

    # result : [frame#, class, error, tx, ty, tz]
    '''
    result = np.transpose(np.array([[data_class_list],
              [error_degree_list],
              [tx_list],
              [ty_list],
              [tz_list]]))
    result = np.reshape(result, (row_size, 5))
    '''
    result = {'frame':index_list, 'class':data_class_list, 'error':error_degree_list, 'tx':tx_list, 'ty':ty_list, 'tz':tz_list}
    result = pd.DataFrame(result)
    result = result.set_index('frame')
    return result

def save_data_to_csv(pose_path, save_path):
    result = generate_data(pose_path)
    #df = pd.DataFrame(result)
    #df.to_csv(save_path, header=None)
    result.to_csv(save_path, header=None)


if __name__ == "__main__":
    #result = gen_degree_tx_ty_tz(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/00.txt")
    #print(result[0,1])
    #print("Row size: ", np.size(result, 0))
    '''
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/00.txt",
                  save_path="./result_csv/00.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/01.txt",
                  save_path="./result_csv/01.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/02.txt",
                  save_path="./result_csv/02.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/03.txt",
                  save_path="./result_csv/03.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/04.txt",
                  save_path="./result_csv/04.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/05.txt",
                  save_path="./result_csv/05.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/06.txt",
                  save_path="./result_csv/06.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/07.txt",
                  save_path="./result_csv/07.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/08.txt",
                  save_path="./result_csv/08.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/09.txt",
                  save_path="./result_csv/09.csv")
    save_data_to_csv(pose_path="D:\\smartcar\\KITTI odometry\\dataset_color\\poses/10.txt",
                  save_path="./result_csv/10.csv")
    '''

    '''
    path = os.path.join('result_csv')
    loaded = np.loadtxt(fname=os.path.join(path, "00.csv"), delimiter=",")
    #trainX = loaded[:,2]
    #print(trainX)
    trainY = loaded[:,1]
    print(trainY)
    print(np.shape(trainY))
    '''

    print(os.path.join('result_csv', '*.csv'))
    filenames = glob.glob(os.path.join('result_csv', '*.csv'))
    print(filenames)