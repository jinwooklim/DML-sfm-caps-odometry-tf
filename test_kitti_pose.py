from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from SfMLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM
from kitti_eval.pose_evaluation_utils import dump_yawclass_pose_seq_TUM
from config import cfg
import pandas as pd
'''
flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
#flags.DEFINE_integer("seq_length", 5, "Sequence length for each example")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
cfg = flags.cfg
'''

def load_image_sequence(dataset_dir, 
                        frames, 
                        tgt_idx, 
                        seq_length, 
                        img_height, 
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = scipy.misc.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq = curr_img
        else:
            image_seq = np.hstack((image_seq, curr_img))
    return image_seq

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

def main():
    sfm = SfMLearner()
    sfm.setup_inference(cfg.img_height, cfg.img_width, 'pose', cfg.seq_length)
    #saver = tf.train.Saver([var for var in tf.trainable_variables()]) 
    saver = tf.train.Saver() 
    #if not os.path.isdir(cfg.output_dir):
    #    os.makedirs(cfg.output_dir)
    seq_dir = os.path.join(cfg.dataset_dir, 'sequences', '%.2d' % cfg.test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (cfg.test_seq, n) for n in range(N)]
    with open(cfg.dataset_dir + 'sequences/%.2d/times.txt' % cfg.test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])
    max_src_offset = (cfg.seq_length - 1)//2
    yaw_class_list = []
    tgt_list = []
    with tf.Session() as sess:
        saver.restore(sess, cfg.ckpt_file)
        for tgt_idx in range(N):
            if not is_valid_sample(test_frames, tgt_idx, cfg.seq_length):
                continue
            if tgt_idx % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx, N))
            # TODO: currently assuming batch_size = 1
            image_seq = load_image_sequence(cfg.dataset_dir, 
                                            test_frames, 
                                            tgt_idx, 
                                            cfg.seq_length, 
                                            cfg.img_height, 
                                            cfg.img_width)
            pred = sfm.inference(image_seq[None, :, :, :], sess, mode='pose')
            pred_poses = pred['pose'][0]
            pred_yaw_class = pred['yaw_class'][0]
            # Insert the target pose [0, 0, 0, 0, 0, 0] 
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)
            ##print("pred_poses : " , pred_poses)
            ##print("pred_yaw_class : " , pred_yaw_class)
            yaw_class_list.append(pred_yaw_class)
            tgt_list.append(tgt_idx)
            curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
            #curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset]
            ##print("times : " , curr_times)
            out_file = cfg.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
            dump_pose_seq_TUM(out_file, pred_poses, curr_times)
            #dump_yawclass_pose_seq_TUM(out_file, pred_poses, pred_yaw_class, curr_times)
    result = {'target':tgt_list, 'class':yaw_class_list}
    result = pd.DataFrame(result)
    result = result.set_index('target')
    result.to_csv("result.csv", header=None)

    
main()
