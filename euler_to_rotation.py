import numpy as np
import tensorflow as tf

def yaw_to_rotation(yaw):
    # roll , pitch is zero
    # So we ignore rollMatrix, pitchMatrix
    
    '''
    yawMatrix = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
        ])
    '''

    yawMatrix = np.array([[tf.cos(yaw), -tf.sin(yaw), tf.constant(0.0)],
        [tf.sin(yaw), tf.cos(yaw), 0.0],
        [0.0, 0.0, 0.1]
        ])

    R = yawMatrix
    theta = tf.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * tf.sin(theta))

    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta

    return rx, ry, rz
